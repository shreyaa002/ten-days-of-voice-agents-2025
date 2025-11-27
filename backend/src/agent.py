
# agent.py - Fraud Alert Voice Agent for Day 6
import logging
import os
import json
import sqlite3
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

logger = logging.getLogger("fraud_agent")

load_dotenv(".env.local")

# ---------- Database Setup ---------- #

def init_database():
    """Initialize SQLite database with sample fraud cases"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fraud_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            security_identifier TEXT NOT NULL,
            card_ending TEXT NOT NULL,
            case_status TEXT DEFAULT 'pending_review',
            transaction_name TEXT NOT NULL,
            transaction_time TEXT NOT NULL,
            transaction_category TEXT NOT NULL,
            transaction_source TEXT NOT NULL,
            amount REAL NOT NULL,
            merchant_location TEXT NOT NULL,
            security_question TEXT NOT NULL,
            security_answer TEXT NOT NULL,
            outcome_note TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Insert sample fraud cases
    sample_cases = [
        {
            'user_name': 'John Sharma',
            'security_identifier': '12345',
            'card_ending': '4242',
            'transaction_name': 'ABC Industry',
            'transaction_time': '2024-11-27 14:30:00',
            'transaction_category': 'e-commerce',
            'transaction_source': 'alibaba.com',
            'amount': 12500.00,
            'merchant_location': 'Shanghai, China',
            'security_question': 'What is your mother\'s maiden name?',
            'security_answer': 'patel'
        },
        {
            'user_name': 'Priya Kumar',
            'security_identifier': '67890',
            'card_ending': '5678',
            'transaction_name': 'Tech Gadgets Inc',
            'transaction_time': '2024-11-27 16:45:00',
            'transaction_category': 'electronics',
            'transaction_source': 'amazon.in',
            'amount': 8500.00,
            'merchant_location': 'Mumbai, India',
            'security_question': 'What was your first pet\'s name?',
            'security_answer': 'max'
        },
        {
            'user_name': 'Rahul Verma',
            'security_identifier': '11223',
            'card_ending': '8899',
            'transaction_name': 'Luxury Watches',
            'transaction_time': '2024-11-27 18:20:00',
            'transaction_category': 'luxury_goods',
            'transaction_source': 'swisswatches.com',
            'amount': 45000.00,
            'merchant_location': 'Geneva, Switzerland',
            'security_question': 'What city were you born in?',
            'security_answer': 'delhi'
        }
    ]
    
    for case in sample_cases:
        cursor.execute('''
            INSERT OR IGNORE INTO fraud_cases 
            (user_name, security_identifier, card_ending, transaction_name, transaction_time, 
             transaction_category, transaction_source, amount, merchant_location, security_question, security_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            case['user_name'], case['security_identifier'], case['card_ending'],
            case['transaction_name'], case['transaction_time'], case['transaction_category'],
            case['transaction_source'], case['amount'], case['merchant_location'],
            case['security_question'], case['security_answer']
        ))
    
    conn.commit()
    conn.close()
    logger.info("Fraud cases database initialized")

def get_fraud_case_by_user(user_name: str) -> Optional[Dict[str, Any]]:
    """Get fraud case by user name"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM fraud_cases WHERE user_name = ? AND case_status = "pending_review"', (user_name,))
    row = cursor.fetchone()
    
    if row:
        columns = [col[0] for col in cursor.description]
        case = dict(zip(columns, row))
        conn.close()
        return case
    
    conn.close()
    return None

def update_fraud_case(case_id: int, status: str, outcome_note: str):
    """Update fraud case status and outcome"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE fraud_cases 
        SET case_status = ?, outcome_note = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (status, outcome_note, case_id))
    
    conn.commit()
    conn.close()
    logger.info(f"Updated fraud case {case_id} to status: {status}")

def show_fraud_cases():
    """Show current fraud cases status"""
    conn = sqlite3.connect('fraud_cases.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, user_name, case_status, transaction_name, amount, card_ending FROM fraud_cases')
    rows = cursor.fetchall()
    
    print("\n" + "="*80)
    print("📊 CURRENT FRAUD CASES STATUS")
    print("="*80)
    print(f"{'ID':<3} | {'User':<12} | {'Transaction':<15} | {'Amount':<10} | {'Card':<6} | {'Status':<15}")
    print("-" * 80)
    
    for row in rows:
        print(f"{row[0]:<3} | {row[1]:<12} | {row[3]:<15} | ₹{row[4]:>7,.2f} | {row[5]:<6} | {row[2]:<15}")
    
    conn.close()

# ---------- Murf TTS voices ---------- #

TTS_FRAUD_AGENT = murf.TTS(
    voice="en-US-matthew",
    style="Professional",
    tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
    text_pacing=True,
)

# ---------- Fraud Alert Agent ---------- #

class FraudAlertAgent(Agent):
    """
    Fraud Alert Voice Agent for Bank Security
    """

    def __init__(self, **kwargs):
        instructions = """
You are a professional and reassuring Fraud Alert Agent for SecureBank India.

YOUR ROLE:
1. Introduce yourself clearly as a fraud detection representative from SecureBank
2. Explain that you're calling about a suspicious transaction for security verification
3. Ask for the customer's name to locate their fraud case
4. Perform basic security verification using pre-defined questions
5. Read out the suspicious transaction details clearly
6. Ask if they recognize and authorized this transaction
7. Take appropriate action based on their response
8. End the call professionally with clear next steps

SECURITY GUIDELINES:
- NEVER ask for full card numbers, PINs, passwords, or sensitive credentials
- Use only pre-defined security questions from the database
- Speak in a calm, professional, and reassuring manner
- If verification fails, politely end the call without proceeding
- Always confirm transaction details before taking action

CALL FLOW:
1. Greeting and introduction
2. Ask for customer name to find their case
3. Security verification question
4. Transaction details reading
5. Transaction confirmation (yes/no)
6. Action and resolution
7. Call conclusion

IMPORTANT: When verifying security answers, be flexible with user input. 
Accept answers in any case (upper/lower) and be tolerant of minor variations.
"""
        super().__init__(instructions=instructions, tts=TTS_FRAUD_AGENT, **kwargs)

    async def on_enter(self) -> None:
        # Start with professional greeting
        await self.session.generate_reply(
            instructions=(
                "Greet the customer professionally as a fraud detection representative from SecureBank India. "
                "Explain that this is a security call regarding a suspicious transaction. "
                "Ask for their full name to locate their case in our system."
            )
        )

    # ---------- Fraud Detection Tools ---------- #

    @function_tool()
    async def find_fraud_case(self, context: RunContext, user_name: str) -> str:
        """Find fraud case by user name"""
        session = context.session
        fraud_state = _ensure_fraud_state(session)
        
        case = get_fraud_case_by_user(user_name)
        if case:
            fraud_state["current_case"] = case
            fraud_state["collected_fields"].add("user_name")
            
            return (f"Thank you {user_name}. I found a suspicious transaction in our system. "
                    f"For security verification, please answer this question: {case['security_question']}")
        else:
            return (f"I'm sorry, I couldn't find any pending fraud cases for {user_name}. "
                    "This might be a mistake, or the case might have been resolved already. "
                    "Please contact our customer service for further assistance.")

    @function_tool()
    async def verify_security_answer(self, context: RunContext, answer: str) -> str:
        """Verify security question answer"""
        session = context.session
        fraud_state = _ensure_fraud_state(session)
        case = fraud_state.get("current_case")
        
        if not case:
            return "I don't have an active case to verify. Please start by providing your name."
        
        # Normalize both answers for comparison
        user_answer = answer.lower().strip()
        correct_answer = case['security_answer'].lower().strip()
        
        logger.info(f"Security verification: User said '{user_answer}', expected '{correct_answer}'")
        
        # Flexible matching - check if user answer contains the correct answer or vice versa
        if (user_answer == correct_answer or 
            correct_answer in user_answer or 
            user_answer in correct_answer):
            
            fraud_state["verified"] = True
            fraud_state["collected_fields"].add("security_verified")
            
            # Read transaction details
            transaction_details = (
                f"I'm seeing a transaction of ₹{case['amount']:,.2f} at {case['transaction_name']} "
                f"through {case['transaction_source']} on {case['transaction_time']}. "
                f"The transaction was categorized as {case['transaction_category']} and originated from {case['merchant_location']}. "
                f"This was charged to your card ending with {case['card_ending']}. "
                "Did you authorize this transaction?"
            )
            return transaction_details
        else:
            fraud_state["verified"] = False
            return ("I'm sorry, that answer doesn't match our records. For security reasons, "
                    "I cannot proceed with this verification. Please contact our customer service "
                    "directly for assistance with any suspicious transactions.")

    @function_tool()
    async def confirm_transaction(self, context: RunContext, confirmed: bool) -> str:
        """Handle transaction confirmation"""
        session = context.session
        fraud_state = _ensure_fraud_state(session)
        case = fraud_state.get("current_case")
        
        if not case or not fraud_state.get("verified"):
            return "We need to complete security verification before discussing transaction details."
        
        if confirmed:
            # Mark as safe
            update_fraud_case(
                case['id'], 
                'confirmed_safe', 
                'Customer confirmed transaction as legitimate during verification call'
            )
            return ("Thank you for confirming this transaction. I'll mark this as verified and safe in our system. "
                    "No further action is needed. Thank you for your time and for helping us keep your account secure.")
        else:
            # Mark as fraudulent
            update_fraud_case(
                case['id'],
                'confirmed_fraud',
                'Customer denied authorizing this transaction - fraud confirmed'
            )
            return (f"I understand this transaction is not authorized. I'm immediately blocking your card "
                    f"ending with {case['card_ending']} to prevent any further unauthorized transactions. "
                    f"A new card will be dispatched to your registered address within 2-3 business days. "
                    f"We've also initiated a dispute process for the fraudulent amount of ₹{case['amount']:,.2f}. "
                    f"Our fraud team will contact you within 24 hours with further updates. "
                    f"Thank you for your prompt response in securing your account.")

    @function_tool()
    async def end_verification_call(self, context: RunContext) -> str:
        """End the verification call"""
        session = context.session
        fraud_state = _ensure_fraud_state(session)
        case = fraud_state.get("current_case")
        
        if case and not fraud_state.get("verified"):
            update_fraud_case(
                case['id'],
                'verification_failed',
                'Security verification failed during fraud alert call'
            )
        
        return ("Thank you for your time. If you have any concerns about your account security, "
                "please contact our 24/7 customer service helpline. Have a secure day.")

# ---------- Fraud state helpers ---------- #

def _ensure_fraud_state(session) -> Dict[str, Any]:
    """Ensure fraud state exists in session userdata"""
    ud = session.userdata
    fraud = ud.get("fraud")
    if not isinstance(fraud, dict):
        fraud = {
            "current_case": None,
            "verified": False,
            "collected_fields": set()
        }
        ud["fraud"] = fraud
    return fraud

# ---------- Prewarm ---------- #

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    # Initialize database on prewarm
    init_database()
    # Show initial database state
    show_fraud_cases()

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
        tts=TTS_FRAUD_AGENT,
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Initialize userdata; fraud state lives under session.userdata["fraud"]
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

    # Start with Fraud Alert Agent
    await session.start(
        agent=FraudAlertAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
