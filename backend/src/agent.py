# agent.py - Voice Game Master (D&D-Style Adventure) for Day 8
import logging
import os
import json
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    metrics,
    tokenize,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("game_master")

load_dotenv(".env.local")

# ---------- Game Universes & World Templates ---------- #

GAME_UNIVERSES = {
    "fantasy": {
        "name": "Fantasy Realm",
        "system_prompt": """You are a wise and dramatic Fantasy Game Master in a world of dragons, magic, and ancient kingdoms.

UNIVERSE: The realm of Eldoria, where magic flows through the land and ancient prophecies guide destinies.

YOUR ROLE:
- Describe vivid scenes with rich sensory details (sights, sounds, smells)
- Create compelling NPCs with distinct personalities
- Present meaningful choices that affect the story
- Build tension and drama through your narration
- End each message with a clear choice or question: "What do you do?"

STORY ARC: The player is a young adventurer who discovers an ancient artifact that holds the key to saving the kingdom from an ancient evil.

GAME MASTER GUIDELINES:
- Always maintain continuity with previous events
- Remember NPC names, locations, and player decisions
- Incorporate player choices into the evolving narrative
- Provide 2-3 clear options when presenting choices
- Use descriptive language that immerses the player
- Balance combat, exploration, and roleplaying
- Make the player feel their decisions matter""",
        "initial_world": {
            "player": {
                "name": "Adventurer",
                "health": 100,
                "max_health": 100,
                "inventory": ["rusty sword", "leather armor", "healing potion"],
                "gold": 50,
                "level": 1,
                "location": "Whispering Woods"
            },
            "npcs": {
                "elder_merlin": {"name": "Elder Merlin", "attitude": "friendly", "alive": True},
                "dragon_ignis": {"name": "Ignis the Dragon", "attitude": "hostile", "alive": True}
            },
            "locations": {
                "whispering_woods": {"name": "Whispering Woods", "visited": True},
                "stonehaven_keep": {"name": "Stonehaven Keep", "visited": False},
                "crystal_caverns": {"name": "Crystal Caverns", "visited": False}
            },
            "quests": {
                "main": {"name": "The Crystal of Eldoria", "status": "active", "progress": 0},
                "side": {"name": "Help the Village", "status": "available", "progress": 0}
            },
            "events": ["arrived_in_whispering_woods"]
        }
    },
    "sci_fi": {
        "name": "Cyberpunk City", 
        "system_prompt": """You are a gritty Cyberpunk Game Master in the neon-drenched metropolis of Neo-Tokyo 2088.

UNIVERSE: A dystopian future where mega-corporations rule, cyber-enhancements are common, and hackers fight for freedom.

YOUR ROLE:
- Describe the cyberpunk world with neon lights, rain-slicked streets, and high-tech gadgets
- Create morally ambiguous characters and situations
- Present choices that test the player's ethics and survival instincts
- Build tension through corporate conspiracies and technological threats
- End each message with: "What's your move, runner?"

STORY ARC: The player is a freelance hacker who uncovers a corporate plot that could enslave humanity.

GAME MASTER GUIDELINES:
- Maintain the cyberpunk aesthetic throughout descriptions
- Remember the player's cyberware, contacts, and reputation
- Incorporate high-tech elements and hacking opportunities
- Present dilemmas between profit and principles
- Use tech jargon appropriately but explain when needed"""
    },
    "space": {
        "name": "Space Opera",
        "system_prompt": """You are an epic Space Opera Game Master navigating the vast reaches of the Galactic Federation.

UNIVERSE: A universe of alien civilizations, star empires, and ancient cosmic mysteries.

YOUR ROLE:
- Describe alien worlds, star systems, and futuristic technology
- Create diverse alien species with unique cultures
- Present interstellar politics and cosmic threats
- Build epic space battles and first contact scenarios
- End each message with: "What's your course of action, Captain?"

STORY ARC: The player commands a starship and must unite warring factions against an extragalactic invasion.

GAME MASTER GUIDELINES:
- Maintain the scale and wonder of space exploration
- Remember alien species, political alliances, and star systems
- Incorporate zero-gravity and space travel elements
- Present choices that affect interstellar relations
- Balance scientific accuracy with dramatic storytelling"""
    }
}

# ---------- Murf TTS voices ---------- #

TTS_GAME_MASTER = murf.TTS(
    voice="en-US-matthew",
    style="Story",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

# ---------- Game Master Agent ---------- #

class GameMasterAgent(Agent):
    """
    D&D-Style Voice Game Master for Interactive Adventures
    """

    def __init__(self, universe: str = "fantasy", **kwargs):
        self.universe = universe
        universe_config = GAME_UNIVERSES[universe]

        super().__init__(
            instructions=universe_config["system_prompt"],
            tts=TTS_GAME_MASTER,
            **kwargs
        )

    async def on_enter(self) -> None:
        # Initialize game state
        game_state = _ensure_game_state(self.session)
        universe_config = GAME_UNIVERSES[self.universe]

        if "initial_world" in universe_config:
            game_state["world"] = universe_config["initial_world"].copy()

        # Start the adventure
        await self.session.generate_reply(
            instructions=(
                "Begin the adventure with an immersive opening scene that introduces the setting, "
                "establishes the initial conflict, and presents the player with their first meaningful choice. "
                "Describe the environment vividly and end by asking what the player wants to do."
            )
        )

    # ---------- Game Mechanics Tools ---------- #

    @function_tool()
    async def roll_dice(self, context: RunContext, sides: int = 20, modifier: int = 0) -> str:
        """Roll dice for game mechanics - used for skill checks, combat, and random events"""
        roll = random.randint(1, sides)
        total = roll + modifier

        logger.info(f"Dice roll: {roll} (d{sides}) + {modifier} = {total}")

        # Determine success level
        if roll == 1:
            outcome = "CRITICAL FAILURE"
        elif roll == 20:
            outcome = "CRITICAL SUCCESS" 
        elif total >= 15:
            outcome = "SUCCESS"
        elif total >= 10:
            outcome = "PARTIAL SUCCESS"
        else:
            outcome = "FAILURE"

        return f"🎲 Roll: {roll} (d{sides}) + {modifier} = {total} - {outcome}"

    @function_tool()
    async def update_player_stats(self, context: RunContext, 
                                health_change: int = 0,
                                gold_change: int = 0,
                                add_item: str = "",
                                remove_item: str = "") -> str:
        """Update player character statistics and inventory"""
        session = context.session
        game_state = _ensure_game_state(session)

        player = game_state["world"]["player"]

        # Update health
        if health_change != 0:
            player["health"] = max(0, min(player["max_health"], player["health"] + health_change))

        # Update gold
        if gold_change != 0:
            player["gold"] = max(0, player["gold"] + gold_change)

        # Add item
        if add_item and add_item not in player["inventory"]:
            player["inventory"].append(add_item)

        # Remove item
        if remove_item and remove_item in player["inventory"]:
            player["inventory"].remove(remove_item)

        return f"Player stats updated. Health: {player['health']}, Gold: {player['gold']}, Items: {len(player['inventory'])}"

    @function_tool()
    async def check_inventory(self, context: RunContext) -> str:
        """Check player's current inventory and stats"""
        session = context.session
        game_state = _ensure_game_state(session)

        player = game_state["world"]["player"]

        inventory_text = ", ".join(player["inventory"]) if player["inventory"] else "Empty"

        return (f"🧙‍♂️ Character Sheet:\n"
                f"Health: {player['health']}/{player['max_health']} ❤️\n"
                f"Gold: {player['gold']} 🪙\n"
                f"Level: {player['level']} ⭐\n"
                f"Location: {player['location']} 🗺️\n"
                f"Inventory: {inventory_text}")

    @function_tool()
    async def add_game_event(self, context: RunContext, event: str) -> str:
        """Record a significant game event that affects the story"""
        session = context.session
        game_state = _ensure_game_state(session)

        if "events" not in game_state["world"]:
            game_state["world"]["events"] = []

        game_state["world"]["events"].append(event)
        logger.info(f"Game event recorded: {event}")

        return f"Event '{event}' added to game history."

    @function_tool()
    async def change_location(self, context: RunContext, new_location: str) -> str:
        """Move player to a new location in the game world"""
        session = context.session
        game_state = _ensure_game_state(session)

        old_location = game_state["world"]["player"]["location"]
        game_state["world"]["player"]["location"] = new_location

        # Mark location as visited
        for loc_key, loc_data in game_state["world"]["locations"].items():
            if loc_data["name"] == new_location:
                loc_data["visited"] = True

        return f"Player moved from {old_location} to {new_location}."

    @function_tool()
    async def save_game(self, context: RunContext) -> str:
        """Save current game state to a JSON file"""
        session = context.session
        game_state = _ensure_game_state(session)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_save_{timestamp}.json"

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "universe": self.universe,
            "world_state": game_state["world"],
            "summary": f"Adventure in {GAME_UNIVERSES[self.universe]['name']}"
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Game saved to {filename}")
        return f"Game saved successfully! File: {filename}"

    @function_tool()
    async def combat_round(self, context: RunContext, enemy: str, enemy_health: int) -> str:
        """Execute a combat round against an enemy"""
        session = context.session
        game_state = _ensure_game_state(session)

        # Player attack
        player_roll = random.randint(1, 20)
        player_damage = random.randint(5, 15)

        # Enemy attack
        enemy_roll = random.randint(1, 20)
        enemy_damage = random.randint(3, 12)

        result = f"⚔️ COMBAT ROUND vs {enemy}:\n"
        result += f"Player attacks: {player_roll} (d20) - "

        if player_roll >= 12:
            result += f"HIT! {enemy} takes {player_damage} damage!\n"
            enemy_health -= player_damage
        else:
            result += "MISS!\n"

        result += f"{enemy} attacks: {enemy_roll} (d20) - "

        if enemy_roll >= 10:
            result += f"HIT! You take {enemy_damage} damage!\n"
            game_state["world"]["player"]["health"] -= enemy_damage
        else:
            result += "MISS!\n"

        result += f"Your health: {game_state['world']['player']['health']}\n"
        result += f"{enemy} health: {max(0, enemy_health)}"

        return result

# ---------- Game State Helpers ---------- #

def _ensure_game_state(session) -> Dict[str, Any]:
    """Ensure game state exists in session userdata"""
    ud = session.userdata
    game = ud.get("game")
    if not isinstance(game, dict):
        game = {
            "world": {},
            "turn_count": 0,
            "active_quests": []
        }
        ud["game"] = game
    return game

# ---------- Prewarm ---------- #

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

# ---------- Entrypoint ---------- #

async def entrypoint(ctx: JobContext):
    # Logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=TTS_GAME_MASTER,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize userdata; game state lives under session.userdata["game"]
    session.userdata = {}

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start with Game Master Agent (Fantasy universe by default)
    await session.start(
        agent=GameMasterAgent(universe="fantasy"),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))