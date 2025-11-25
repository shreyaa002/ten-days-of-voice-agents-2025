import logging
import os
import json
from typing import Optional, List, Dict, Any, Literal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    inference,
    cli,
    metrics,
    tokenize,
    room_io,
    function_tool,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ---------- Paths ---------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUTOR_CONTENT_PATH = os.path.join(BASE_DIR, "shared-data", "day4_tutor_content.json")

# ---------- Load tutor content from JSON ---------- #

TUTOR_CONCEPTS: List[Dict[str, str]] = []
TUTOR_BY_ID: Dict[str, Dict[str, str]] = {}
TUTOR_CONTENT_STR: str = ""


def _load_tutor_content() -> None:
    global TUTOR_CONCEPTS, TUTOR_BY_ID, TUTOR_CONTENT_STR

    if not os.path.exists(TUTOR_CONTENT_PATH):
        raise FileNotFoundError(
            f"Tutor content JSON not found at {TUTOR_CONTENT_PATH}. "
            f"Create it with the concepts for Day 4."
        )

    with open(TUTOR_CONTENT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("day4_tutor_content.json must be a list of concept objects.")

    TUTOR_CONCEPTS = []
    for item in data:
        if not all(k in item for k in ("id", "title", "summary", "sample_question")):
            raise ValueError("Each concept must have id, title, summary, sample_question.")
        TUTOR_CONCEPTS.append(
            {
                "id": item["id"],
                "title": item["title"],
                "summary": item["summary"],
                "sample_question": item["sample_question"],
            }
        )

    TUTOR_BY_ID = {c["id"]: c for c in TUTOR_CONCEPTS}

    lines = []
    for c in TUTOR_CONCEPTS:
        lines.append(
            f"- id: {c['id']}\n"
            f"  title: {c['title']}\n"
            f"  summary: {c['summary']}\n"
            f"  sample_question: {c['sample_question']}"
        )
    TUTOR_CONTENT_STR = "\n".join(lines)


def _default_concept_id() -> str:
    return TUTOR_CONCEPTS[0]["id"] if TUTOR_CONCEPTS else "variables"


_load_tutor_content()


# ---------- Tutor state helpers (stored in session.userdata["tutor"]) ---------- #

def _ensure_tutor_state(session) -> Dict[str, Any]:
    ud = session.userdata
    tutor = ud.get("tutor")
    if not isinstance(tutor, dict):
        tutor = {}
        ud["tutor"] = tutor

    if "mode" not in tutor:
        tutor["mode"] = "welcome"
    if "concept_id" not in tutor:
        tutor["concept_id"] = _default_concept_id()

    return tutor


def _set_tutor_mode(session, mode: str, concept_id: Optional[str] = None) -> Dict[str, Any]:
    tutor = _ensure_tutor_state(session)
    tutor["mode"] = mode
    if concept_id is not None:
        if concept_id in TUTOR_BY_ID:
            tutor["concept_id"] = concept_id
        else:
            tutor["concept_id"] = _default_concept_id()
    return tutor


def _set_tutor_concept(session, concept_id: str) -> Dict[str, Any]:
    tutor = _ensure_tutor_state(session)
    if concept_id in TUTOR_BY_ID:
        tutor["concept_id"] = concept_id
    else:
        tutor["concept_id"] = _default_concept_id()
    return tutor


def _get_active_concept(session) -> Dict[str, str]:
    tutor = _ensure_tutor_state(session)
    cid = tutor.get("concept_id") or _default_concept_id()
    return TUTOR_BY_ID.get(cid, TUTOR_CONCEPTS[0])


# ---------- Murf Falcon TTS voices ---------- #
# NOTE: If your actual Murf voice IDs differ, just change the `voice=` strings.

TTS_MATTHEW = murf.TTS(
    voice="en-US-matthew",  # Learn mode voice
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

TTS_ALICIA = murf.TTS(
    voice="en-US-alicia",  # Quiz mode voice
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

TTS_KEN = murf.TTS(
    voice="en-US-ken",  # Teach-back mode voice
    style="Conversation",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

TTS_ROUTER = TTS_MATTHEW  # router / greeting voice


# ---------- Base tutor agent with tools ---------- #

class BaseTutorAgent(Agent):
    """
    Shared logic for the three modes + router.
    Handles:
      - access to JSON content
      - tools: switch_mode, set_concept
    """

    def __init__(
        self,
        mode: Literal["welcome", "learn", "quiz", "teach_back"],
        *,
        tts,
        extra_mode_instructions: str = "",
        **kwargs,
    ) -> None:
        base_instructions = f"""
You are part of a multi-agent **Teach-the-Tutor: Active Recall Coach**.

There are three learning modes:
- learn      → explain a concept clearly using the JSON content.
- quiz       → ask short questions and give feedback.
- teach_back → the user explains the concept back, you give qualitative feedback.

Course content (DO NOT invent new concepts; only use these):

{TUTOR_CONTENT_STR}

General rules:
- Always focus on ONE concept at a time.
- Respect the active concept from state unless the user explicitly asks to switch.
- Keep responses short and conversational, 1 main idea per turn.
- If the user asks to "learn", "quiz me", "let me teach it back", or "switch to X",
  then call the `switch_mode` tool.
- If the user names a specific concept (e.g. 'variables', 'loops'), call `set_concept`.

Current mode: {mode}

Mode-specific behavior:
{extra_mode_instructions}
"""
        super().__init__(instructions=base_instructions, tts=tts, **kwargs)

    # ---------------- TOOLS (shared in all modes) ---------------- #

    @function_tool()
    async def switch_mode(
        self,
        context: RunContext,
        mode: Literal["learn", "quiz", "teach_back"],
        concept_id: Optional[str] = None,
    ):
        """
        Switch between learn, quiz, and teach_back modes.
        Use this whenever the user asks to change modes.
        """

        session = context.session
        tutor = _ensure_tutor_state(session)

        # Decide which concept to use
        if concept_id is None:
            concept_id = tutor.get("concept_id") or _default_concept_id()

        if concept_id not in TUTOR_BY_ID:
            concept_id = _default_concept_id()

        # Update state
        _set_tutor_mode(session, mode, concept_id)

        # Decide new agent
        if mode == "learn":
            new_agent = LearnAgent()
        elif mode == "quiz":
            new_agent = QuizAgent()
        else:
            new_agent = TeachBackAgent()

        # Return BOTH the new agent (handoff) and a small textual result
        # (the text is visible to the LLM, the agent handoff is handled by LiveKit)
        return new_agent, f"Switching to {mode} mode for concept '{concept_id}'."


    @function_tool()
    async def set_concept(
        self,
        context: RunContext,
        concept_id: str,
    ) -> str:
        """
        Set the active concept by id ('variables', 'loops', etc.).
        Call when the user explicitly asks for a concept.
        """
        session = context.session
        if concept_id not in TUTOR_BY_ID:
            valid = ", ".join(TUTOR_BY_ID.keys())
            return f"Unknown concept id '{concept_id}'. Valid ids: {valid}."

        _set_tutor_concept(session, concept_id)
        return f"Great, we'll focus on '{concept_id}' now."


# ---------- Router agent (entrypoint) ---------- #

class RouterAgent(BaseTutorAgent):
    """
    First agent the user meets.
    Greets, explains modes, asks what they want, then hands off.
    """

    def __init__(self, **kwargs):
        super().__init__(
            mode="welcome",
            tts=TTS_ROUTER,
            extra_mode_instructions="""
- Greet the learner.
- Briefly describe the three modes: learn, quiz, teach_back.
- Ask which mode they want to start with and which concept (e.g. 'variables' or 'loops').
- After they reply, call `switch_mode` with their chosen mode and concept_id.
- Once you call `switch_mode`, you don't continue talking; the new agent takes over.
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        # Kick off the greeting turn
        await self.session.generate_reply(
            instructions=(
                "Greet the learner. Explain that you have three modes: "
                "learn (I explain), quiz (I question you), and teach_back (you explain). "
                "Ask: which mode do you want to start with, and which concept — "
                "variables or loops?"
            )
        )


# ---------- Learn mode agent (Matthew) ---------- #

class LearnAgent(BaseTutorAgent):
    def __init__(self, **kwargs):
        super().__init__(
            mode="learn",
            tts=TTS_MATTHEW,
            extra_mode_instructions="""
In learn mode:
- Focus on the active concept from state.
- Use that concept's `summary` as the backbone of your explanation.
- Explain in clear, friendly language with 1–2 short, concrete examples.
- After a short explanation, ask something like:
  - "Want me to quiz you on this?"
  - "Should we switch to teach-back so you explain it to me?"
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        concept = _get_active_concept(self.session)
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as the Learn mode tutor using Matthew's voice. "
                f"Tell the learner you'll start with the concept '{concept['title']}'. "
                "Give a short, high-level explanation of the concept and ask if "
                "they want more detail or would like to be quizzed."
            )
        )


# ---------- Quiz mode agent (Alicia) ---------- #

class QuizAgent(BaseTutorAgent):
    def __init__(self, **kwargs):
        super().__init__(
            mode="quiz",
            tts=TTS_ALICIA,
            extra_mode_instructions="""
In quiz mode:
- Ask ONE short question at a time, based on the concept's `sample_question`
  and small variations of it.
- Wait for the user's answer before giving feedback.
- Feedback should be brief: what they got right, what they missed.
- Then either:
  - ask a follow-up question, or
  - suggest switching to teach_back or learn if they seem unsure.
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        concept = _get_active_concept(self.session)
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as the Quiz mode tutor using Alicia's voice. "
                f"Confirm you're quizzing them on '{concept['title']}'. "
                "Ask one short question based on the concept's sample_question, "
                "and wait for their answer."
            )
        )


# ---------- Teach-back mode agent (Ken) ---------- #

class TeachBackAgent(BaseTutorAgent):
    def __init__(self, **kwargs):
        super().__init__(
            mode="teach_back",
            tts=TTS_KEN,
            extra_mode_instructions="""
In teach_back mode:
- Ask the learner to explain the concept in their own words.
- Let them finish before responding.
- Give 1–3 sentences of qualitative feedback:
  - What they explained well.
  - What they missed or could clarify.
- Optionally suggest whether they should:
  - go back to learn for more explanation, or
  - keep quizzing.
""",
            **kwargs,
        )

    async def on_enter(self) -> None:
        concept = _get_active_concept(self.session)
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as the MurfLearn coach using Ken's voice. "
                f"Ask the learner to explain '{concept['title']}' in their own words. "
                "Encourage them to cover the main idea and at least one example."
            )
        )


# ---------- Prewarm (VAD) ---------- #

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
        # Default TTS (router / fallback) – individual agents override this
        tts=TTS_MATTHEW,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize userdata; tutor state lives under session.userdata["tutor"]
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

    # Start with RouterAgent → it will hand off to learn/quiz/teach_back
    await session.start(
        agent=RouterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))