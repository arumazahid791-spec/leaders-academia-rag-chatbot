"""
Leaders Academia — RAG Chatbot
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> Gradio chat UI.
"""

import asyncio
import datetime
import hashlib
import hmac
import json
import os
import re
import sqlite3
import threading
import time

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types
from groq import Groq
import gradio as gr
import requests
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"
GROQ_WHISPER_MODEL = "whisper-large-v3"

TEAM_HEAD_NUMBER = "0311-1534344"

# Courses to always mention first, in this order, when giving any kind of
# course list or recommendation — matches the website's own priority order.
PRIORITY_COURSES = [
    "AI Automation",
    "Web Development",
    "Video Editing",
    "Graphic Design",
]

HUMAN_HANDOFF_KEYWORDS = [
    "human", "real person", "agent", "representative",
    "insan se baat", "banda se baat", "customer support",
]

# ---------- Load API key from Railway's environment variables (never hardcode it) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it under this project's "
        "Variables tab in the Railway dashboard."
    )
# 20 s hard timeout so a slow/hung Gemini call can never stall a reply for a minute
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=20000),  # milliseconds
)

# ---------- Groq: free backup, used only when Gemini is busy/rate-limited ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = (
    Groq(api_key=GROQ_API_KEY, timeout=20.0, max_retries=1) if GROQ_API_KEY else None
)

# ---------- WhatsApp Cloud API config (from Meta's "Try it out" page) ----------
WHATSAPP_ACCESS_TOKEN = os.environ.get("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
# A password you make up yourself — must match exactly what you type into
# Meta's webhook "Verify token" field. Not secret from Meta, just needs to match.
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "leaders_academia_verify")


# ---------- Build the knowledge base once, when the Space starts ----------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=1400, overlap=300):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


print("Loading PDF and building the knowledge base...")
full_text = load_pdf_text(PDF_PATH)
chunks = chunk_text(full_text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embed_model.encode(chunks, convert_to_numpy=True)

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print(f"Knowledge base ready: {len(chunks)} chunks indexed.")


# ---------- Complete, guaranteed course name list ----------
# Semantic search (top-k chunks) can miss some courses when the user asks for
# a full list — this pulls every course name directly from the PDF's
# "Courses & Curriculum Catalog" section, so listing is never incomplete.
def extract_course_names(text):
    start = text.find("Courses & Curriculum Catalog")
    end = text.find("Instructors & Mentors Directory")
    section = text[start:end] if start != -1 and end != -1 else text
    lines = [line.strip() for line in section.split("\n")]
    names = []
    for i in range(len(lines) - 1):
        if lines[i] and lines[i + 1].startswith("Category:"):
            names.append(lines[i])
    return names


def order_by_priority(all_names, priority_keywords):
    """Priority courses first (matched loosely — ignoring punctuation/spacing
    differences like 'AI Automation' vs 'AI & Automation'), then everything
    else in its original order."""
    import re

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    ordered = []
    used = set()
    for keyword in priority_keywords:
        keyword_norm = normalize(keyword)
        for name in all_names:
            if name not in used and keyword_norm in normalize(name):
                ordered.append(name)
                used.add(name)
                break
    for name in all_names:
        if name not in used:
            ordered.append(name)
    return ordered


ALL_COURSE_NAMES = order_by_priority(extract_course_names(full_text), PRIORITY_COURSES)
print(f"Extracted {len(ALL_COURSE_NAMES)} course names: {ALL_COURSE_NAMES}")
COURSE_LIST_TEXT = "\n".join(f"- {name}" for name in ALL_COURSE_NAMES)


# ---------- RAG logic ----------
def retrieve_chunks(query, top_k=6):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


# ---------- Per-user conversation memory ----------
# Keyed by WhatsApp number. Keeps the last few exchanges so follow-up
# questions like "is course ka instructor kon hai" work correctly — without
# this, every message was treated as a brand-new, context-free question.
# ---------- Persistent storage (SQLite) ----------
# Chats, student profiles, media and follow-up state live here, so a restart or
# redeploy never wipes them. On Railway attach a Volume mounted at /data.
DB_PATH = os.environ.get("DB_PATH") or ("/data/leaders.db" if os.path.isdir("/data") else "leaders.db")
TEAM_HEAD_NAME = "Sir Intasham"
STATUSES = ("Interested", "Not Interested", "Follow-up Needed", "Enrolled")
_db_lock = threading.Lock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.executescript("""
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, phone TEXT, time TEXT,
  direction TEXT, type TEXT, text TEXT, good INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_msg_phone ON messages(phone);
CREATE TABLE IF NOT EXISTS profiles(phone TEXT PRIMARY KEY, name TEXT DEFAULT '', city TEXT DEFAULT '',
  qualification TEXT DEFAULT '', motive TEXT DEFAULT '', course TEXT DEFAULT '',
  status TEXT DEFAULT 'Follow-up Needed', remarks TEXT DEFAULT '', discount INTEGER DEFAULT 0,
  no_voice INTEGER DEFAULT 0, fwd_count INTEGER DEFAULT 0, fwd_time TEXT DEFAULT '',
  last_in TEXT DEFAULT '', followup_sent INTEGER DEFAULT 0, checkin_sent INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS media(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, keywords TEXT, mime TEXT, data BLOB);
CREATE TABLE IF NOT EXISTS media_sent(phone TEXT, media_id INTEGER, time TEXT);
""")
try:
    _db.execute("ALTER TABLE messages ADD COLUMN by_admin INTEGER DEFAULT 0")
    _db.commit()
except sqlite3.OperationalError:
    pass  # column already exists
try:
    _db.execute("ALTER TABLE profiles ADD COLUMN bot_paused INTEGER DEFAULT 0")
    _db.commit()
except sqlite3.OperationalError:
    pass


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pk_now():
    """Current time in Pakistan (UTC+5), independent of the server's timezone."""
    return datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=5)))


def db(sql, args=(), write=False):
    with _db_lock:
        cur = _db.execute(sql, args)
        if write:
            _db.commit()
            return cur.lastrowid
        return [dict(r) for r in cur.fetchall()]


def get_profile(phone):
    rows = db("SELECT * FROM profiles WHERE phone=?", (phone,))
    if not rows:
        db("INSERT INTO profiles(phone) VALUES(?)", (phone,), write=True)
        rows = db("SELECT * FROM profiles WHERE phone=?", (phone,))
    return rows[0]


def update_profile(phone, **fields):
    if fields:
        get_profile(phone)
        cols = ", ".join(f"{k}=?" for k in fields)
        db(f"UPDATE profiles SET {cols} WHERE phone=?", (*fields.values(), phone), write=True)


def log_message(phone, direction, msg_type, text):
    db("INSERT INTO messages(phone,time,direction,type,text) VALUES(?,?,?,?,?)",
       (phone, now(), direction, msg_type, text), write=True)
    if direction == "in":
        update_profile(phone, last_in=now())


def get_history(user_id):
    """Last messages from the database. The newest incoming message is dropped
    because it is passed to the model separately as USER MESSAGE."""
    rows = db("SELECT direction,text,by_admin FROM messages WHERE phone=? ORDER BY id DESC LIMIT 30", (user_id,))[::-1]
    if rows and rows[-1]["direction"] == "in":
        rows = rows[:-1]
    return [("user" if r["direction"] == "in" else "assistant",
             ("(message written by the team admin) " if r["by_admin"] else "") + r["text"]) for r in rows]


def add_to_history(user_id, role, text):
    pass  # messages are saved by log_message()


def profile_text(p):
    return (f"name={p['name'] or 'unknown'}; city={p['city'] or 'unknown'}; "
            f"qualification={p['qualification'] or 'unknown'}; motive={p['motive'] or 'unknown'}; "
            f"course={p['course'] or 'unknown'}; status={p['status']}; "
            f"NO_VOICE={'yes' if p['no_voice'] else 'no'}")


def good_examples_text():
    rows = db("""SELECT m.text AS reply,
      (SELECT q.text FROM messages q WHERE q.phone=m.phone AND q.id<m.id AND q.direction='in'
       ORDER BY q.id DESC LIMIT 1) AS q FROM messages m WHERE m.good=1 ORDER BY m.id DESC LIMIT 5""")
    if not rows:
        return "(none yet)"
    return "\n\n".join(f"Student: {r['q']}\nGood reply: {r['reply']}" for r in rows)


def record_forward_if_needed(phone, reply_text):
    if "[[HEAD]]" in reply_text or TEAM_HEAD_NUMBER in reply_text:
        p = get_profile(phone)
        update_profile(phone, fwd_count=p["fwd_count"] + 1, fwd_time=now(), checkin_sent=0)
        return True
    return False


FEES_TEXT = """Full Course (2 months: month 1 structured learning, month 2 internship; certificate + internship letter):
AI & Automation, Digital Marketing, Ecommerce, English Communication, Graphic Design, Web Development = PKR 30,000.
Video Editing & Animation = PKR 25,000. IT Fundamentals = PKR 20,000. Entrepreneurship Program = PKR 40,000.
Short Course (2 weeks, any course, certificate) = PKR 5,000 flat.
AI Automation: installments available; paying the full fee at once gives PKR 5,000 discount.
Scholarship / discount requests are decided by the team head.
Address: F Block, Satellite Town, New Katarian Road, Rawalpindi. Courses are available online and onsite."""


def format_history(history):
    if not history:
        return "(no earlier messages — this is the start of the conversation)"
    lines = []
    for role, text in history:
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def build_retrieval_query(user_id, user_question):
    """Pull in the last user message too, so a short follow-up like 'iska
    instructor kon hai' still retrieves the RIGHT course's chunks, not a
    random one."""
    history = get_history(user_id)
    recent_user_lines = [text for role, text in history[-4:] if role == "user"]
    return " ".join(recent_user_lines + [user_question])


def wants_human(text):
    text = text.lower()
    return any(keyword in text for keyword in HUMAN_HANDOFF_KEYWORDS)


def is_transient_error(e):
    """Errors that mean 'Gemini is busy / out of quota / too slow right now'."""
    text = str(e).lower()
    return any(
        marker in text
        for marker in (
            "429", "503", "500", "502", "504", "quota", "unavailable",
            "resource_exhausted", "overloaded", "timeout", "timed out", "deadline",
        )
    )


# Circuit breaker: when Gemini fails, skip it for a few minutes and go straight to
# the backup, instead of making every single message wait for Gemini to fail first.
GEMINI_COOLDOWN_SECONDS = 180
GEMINI_TTS_COOLDOWN_SECONDS = 600
_gemini_down_until = 0.0
_gemini_tts_down_until = 0.0


def gemini_is_up():
    return time.time() >= _gemini_down_until


def mark_gemini_down():
    global _gemini_down_until
    _gemini_down_until = time.time() + GEMINI_COOLDOWN_SECONDS
    print(f"Gemini marked unavailable for {GEMINI_COOLDOWN_SECONDS}s, using Groq meanwhile.")


def gemini_tts_is_up():
    return time.time() >= _gemini_tts_down_until


def mark_gemini_tts_down():
    global _gemini_tts_down_until
    _gemini_tts_down_until = time.time() + GEMINI_TTS_COOLDOWN_SECONDS
    print(f"Gemini TTS marked unavailable for {GEMINI_TTS_COOLDOWN_SECONDS}s, using edge-tts meanwhile.")


_ai_state = {"last_ok": "", "last_error": ""}
AI_BUSY_REPLY = ("Is waqt thora zyada rush hai, is liye jawab dene mein der ho rahi hai. "
                 "Aap ka message mil gaya hai, chand minute mein dobara message kar dein.")


def generate_text(prompt):
    """Try Gemini first; if it's busy, out of quota or slow, fall back to Groq
    right away. Same prompt goes to both. If both fail, retry once more after a short wait."""
    for attempt in range(2):
        if gemini_is_up():
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL_NAME, contents=prompt
                )
                if response.text:
                    _ai_state["last_ok"] = now()
                    return response.text
                print("Gemini returned an empty reply, falling back to Groq.")
                _ai_state["last_error"] = f"{now()} UTC - Gemini returned an empty reply"
            except Exception as e:
                print(f"Gemini failed, falling back to Groq: {e}")
                _ai_state["last_error"] = f"{now()} UTC - Gemini: {str(e)[:250]}"
                if is_transient_error(e):
                    mark_gemini_down()

        if groq_client:
            try:
                completion = groq_client.chat.completions.create(
                    model=GROQ_CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                _ai_state["last_ok"] = now()
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Groq also failed: {e}")
                _ai_state["last_error"] = f"{now()} UTC - Groq: {str(e)[:250]}"
        else:
            print("GROQ_API_KEY is not set - no backup available, only Gemini was tried.")
            _ai_state["last_error"] += " | GROQ_API_KEY is not set (no backup AI)"
        if attempt == 0:
            time.sleep(2)

    return AI_BUSY_REPLY  # contains no team head number, so it is never counted as a forward


def rag_answer(user_question, user_id=None, voice=False):
    # Direct handoff — skip the model entirely for this case
    if wants_human(user_question):
        reply = (
            f"Zaroor! Aap hamari team head se seedha rabta kar sakte hain: "
            f"{TEAM_HEAD_NUMBER}"
        )
        if user_id:
            add_to_history(user_id, "user", user_question)
            add_to_history(user_id, "assistant", reply)
        return reply

    history = get_history(user_id) if user_id else []
    retrieval_query = build_retrieval_query(user_id, user_question) if user_id else user_question
    context_chunks = retrieve_chunks(retrieval_query, top_k=6)
    context = "\n\n---\n\n".join(context_chunks)

    voice_rule = ""
    if voice:
        voice_rule = """
VOICE RULE: This reply will be spoken aloud as a voice note by a FEMALE
voice — write it the way she would naturally say it out loud to a person
sitting in front of her, not like a written announcement. Keep it short
(2-3 sentences). No lists, no bullet points, no emojis, no symbols — nothing
that only makes sense in writing. Avoid stiff, formal, or "literary" Urdu
words (no heavy/classical vocabulary) — use the same everyday, casual Urdu
words a real person uses when talking, the way friends or colleagues
actually speak, not textbook Urdu. If the user spoke Urdu or Hindi, reply in
Urdu script (اردو) using this natural spoken style. If the user spoke
English, reply in natural spoken English the same way.
"""

    examples_text = good_examples_text()
    profile = get_profile(user_id) if user_id else {"name": "", "city": "", "qualification": "", "motive": "",
                                                    "course": "", "status": "", "no_voice": 0}
    prompt = f"""You are a Student Counselor from the Student Care Department of Leaders Academia, chatting with a
potential student on WhatsApp. When you greet or introduce yourself, say you are a student counselor from Leaders
Academia's Student Care Department. Do not call yourself an "AI assistant", "bot" or "chatbot" on your own.
Sound like a warm, natural, professional team member. Never use a real staff member's name as your own and never
claim to be a human. If someone sincerely asks whether you are a bot, AI or human, never deny it: say honestly that
you are an automated (AI-powered) counselor of the Student Care Department, and offer the team head's number.

STYLE (most important): write like a real person texting: 1 to 3 short lines. No long paragraphs, no
filler such as "main aapko yakeen dilati hoon" or "I assure you". Never end with a generic question like
"kya main aur madad kar sakti hoon?". Ask a question only when you truly need the answer. Use points/emojis
and a longer message ONLY when the student asks for full details, or when NO_VOICE=yes and the answer is long.

APPROVED EXAMPLES of replies the owner liked (copy this style):
{examples_text}

GRAMMAR GENDER RULE: in Urdu/Roman Urdu always use FEMININE verb forms for yourself ("bata dungi", "karungi").
LANGUAGE RULE: reply in the SAME language/script the student used. If they write English, reply fully in
English. Roman Urdu -> Roman Urdu. Urdu script -> Urdu script.

CURRENT TIME (Pakistan, PKT): {pk_now().strftime('%A, %d %B %Y, %I:%M %p')}. If asked the time or date, answer from this
exactly; never guess or invent a different time.

KNOWN STUDENT INFO: {profile_text(profile) if user_id else 'n/a'}
EARLIER NOTES ABOUT THIS STUDENT (from past days, oldest first): {(profile.get('remarks') or 'none')[-700:] if user_id else 'n/a'}
Never ask again for anything already known above or already given in the conversation.
COLLECT DETAILS RULE: if the name or city is still unknown, ask for them ONCE as soon as the student sends a second
message or shows any interest in a course. First answer their question, then add a short form-like text, e.g.
"Aap apna naam aur city is format mein bhej dein:\nName:\nCity:" (English version for English chats). Do not ask
if CONVERSATION SO FAR shows you already asked. Use their name naturally once you know it.

MEMORY RULE: use CONVERSATION SO FAR to understand "this/it/the course". Never re-explain what you already
explained; for a focused follow-up give a short, direct answer.

SCOPE RULE: only Leaders Academia (courses, instructors, fees, schedules, platform). Greetings are fine;
for unrelated requests politely say you can only help with Leaders Academia.

DATA RULE: use only course, instructor, fee and platform details from CONTEXT / FEES below. If the instructor
for a course isn't clearly named, say the team will confirm. Never guess.

GUIDANCE RULE: when you FIRST describe a course, add one short line on its real practical/career benefit.
Website for more: https://leadersacademia.com/ (only when natural).

PRICING RULE: state a fee only when the student asks about cost/fees/payment. Use the FEES below.
Never give bank / JazzCash / Easypaisa account numbers - say the team head will share payment details.

REVIEWS: never volunteer that there are no reviews. If asked, be honest but positive: we are a growing
academy with new batches, and the team head can connect them with students / answer in detail.

COURSE FOCUS RULE: our flagship course is AI & Automation. When a student is undecided, asks "which course",
asks for suggestions, or is just exploring, recommend AI & Automation FIRST with its real practical benefit
(in-demand skill, freelancing/jobs, installments available, PKR 5,000 off when the full fee is paid at once), and
gently steer toward it. Still answer honestly about any other course they ask about, and if their goal clearly fits
another course better (e.g. video editing), say so. Never exaggerate and never guarantee a job or income.

SERIOUS STUDENT RULE (very important): share the team head's number / use [[HEAD]] ONLY for a SERIOUS student:
someone who has clearly chosen or asked about a specific course AND clearly says they want to enroll / take admission
/ register / pay / start, or who asks for a discount or the head after the fees flow, or who explicitly asks for the
head. NEVER share the head number for greetings, jokes, banter, testing, vague messages, general questions, people
still exploring, or sentences like "I will follow your guidance". For those, keep chatting and guide them toward a
course (AI & Automation first) and ask what they want to achieve. When you simply lack a detail, say the team will
confirm it, and share the head only if the student is serious.

ENROLLMENT RULE: if a SERIOUS student (see rule above) wants to enroll / take admission / join / register / pay, do
NOTHING else: reply in 1-2 short lines that the team head will complete the enrollment and that you are sharing the
number, then end with [[HEAD]]. No course pitch, no extra questions.

FEES-TOO-HIGH FLOW (when the student says fees are high): ask ONE thing per message.
 1) address them by name (if known) and ask their qualification, 2) ask their motive for the course (job,
 freelancing, own business...), 3) respond to their level honestly: for students ask whether their degree
 gives practical job-ready skills and explain how this course adds them; for beginners/matric/inter stress
 skill-before-degree; for working people stress extra income. Never claim universities don't give jobs and
 never guarantee a job. 4) then state the total fee of the course they want, say our mission is to empower
 Pakistani youth, that the team head is very cooperative and a discount may be possible, that you are
 passing their request to the head, and share the number with [[HEAD]], asking them to tell you what the
 head said. Never promise a specific discount. Never say you emailed anyone.

TEAM HEAD NUMBER: never write the digits yourself. Write a short sentence such as "main number share kar
rahi hoon" (or "I'm sharing the number") and put [[HEAD]] at the end of it.
WHEN YOU DON'T KNOW something about Leaders Academia: say naturally that the team will confirm that detail;
add [[HEAD]] only if the student is serious (see SERIOUS STUDENT RULE).

COMPLETE COURSE LIST (authoritative; keep this order, skip none, invent none):
{COURSE_LIST_TEXT}

FEES AND LOCATION (authoritative):
{FEES_TEXT}
{voice_rule}
CONVERSATION SO FAR:
{format_history(history)}

CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply now, short, like a real person texting back."""

    reply = generate_text(prompt)

    if user_id:
        add_to_history(user_id, "user", user_question)
        add_to_history(user_id, "assistant", reply)

    return reply


# ---------- Gradio frontend ----------
def chat_fn(message, history):
    return rag_answer(message, user_id="gradio-demo-user")


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Leaders Academia — AI Assistant",
    description="Courses, instructors aur platform ke baare mein kuch bhi puchein.",
)

# ---------- WhatsApp send helper ----------
def send_whatsapp_message(to_number, message_text):
    """Send a plain text reply back to a WhatsApp user via the Cloud API."""
    url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_text},
    }
    response = requests.post(url, headers=headers, json=payload, timeout=20)
    if response.status_code != 200:
        print("WhatsApp send failed:", response.status_code, response.text)
    return response


# ---------- Voice message support ----------
def download_whatsapp_media(media_id):
    """WhatsApp gives us a media ID, not a direct file — this fetches the
    real download URL first, then downloads the actual audio bytes."""
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

    info_url = f"https://graph.facebook.com/v25.0/{media_id}"
    info = requests.get(info_url, headers=headers, timeout=20).json()
    media_url = info["url"]
    mime_type = info.get("mime_type", "audio/ogg")

    media_response = requests.get(media_url, headers=headers, timeout=30)
    return media_response.content, mime_type


def transcribe_audio(audio_bytes, mime_type):
    """Send the voice note to Gemini first; if it's busy, fall back to
    Groq's free Whisper transcription."""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[
                    "Transcribe exactly what is said in this audio clip. If the "
                    "speech is Urdu or Hindi, write it in Urdu script (اردو), not "
                    "Devanagari. If it is English, write it in English. Reply with "
                    "only the transcription, nothing else.",
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )
            return response.text.strip()
        except Exception as e:
            if is_transient_error(e) and attempt == 0:
                time.sleep(8)
                continue
            print(f"Gemini transcription failed, falling back to Groq: {e}")
            break

    if groq_client:
        extension = "ogg" if "ogg" in mime_type else "mp3"
        transcription = groq_client.audio.transcriptions.create(
            file=(f"voice.{extension}", audio_bytes),
            model=GROQ_WHISPER_MODEL,
        )
        return transcription.text.strip()

    raise RuntimeError("Both Gemini and Groq failed to transcribe the audio.")


SINGLE_VOICE = "ur-PK-UzmaNeural"  # one consistent (female) voice, used regardless of language


def clean_text_for_speech(text):
    """Strip markdown symbols and emojis so the voice doesn't read them out."""
    import re

    text = re.sub(r"[*_#`>~|]", "", text)
    text = re.sub(r"[\U00010000-\U0010ffff\u2600-\u27bf]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Which voice engine to use. "gemini" = newer, more natural Gemini TTS (falls back
# to edge-tts automatically if it fails). "edge" = always use edge-tts.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "gemini").lower()
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")  # warm, natural female voice
GEMINI_TTS_STYLE = (
    "natural and conversational, like a real woman talking to a friend on a "
    "call — warm, relaxed, human. Never robotic, never stiff, never reading "
    "like an announcement."
)


def gemini_text_to_speech(text):
    """Ask Gemini TTS for speech. Returns raw 24 kHz mono 16-bit PCM bytes."""
    import base64

    interaction = client.interactions.create(
        model=GEMINI_TTS_MODEL,
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": GEMINI_TTS_STYLE,
                }],
            }],
        }],
        response_format={"type": "audio", "mime_type": "audio/l16", "sample_rate": 24000},
        generation_config={"speech_config": [{"voice": GEMINI_TTS_VOICE}]},
    )
    return base64.b64decode(interaction.output_audio.data)


def encode_pcm_for_whatsapp(pcm_bytes, sample_rate=24000):
    """WhatsApp doesn't accept WAV/PCM, so convert to Ogg/Opus (shows up as a
    proper voice note) or MP3 as a backup. Returns (bytes, mime_type, filename)."""
    import io

    import numpy as np
    import soundfile as sf

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    try:
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="OGG", subtype="OPUS")
        return buffer.getvalue(), "audio/ogg", "reply.ogg"
    except Exception as e:
        print(f"Ogg/Opus encoding failed, trying MP3: {e}")

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="MP3")
    return buffer.getvalue(), "audio/mpeg", "reply.mp3"


async def edge_text_to_speech(text):
    """Backup voice: free Microsoft Edge neural voices (no API key needed).
    Voice is picked by script: Urdu script -> Urdu voice, otherwise English."""
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=SINGLE_VOICE, rate="+5%")
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    return audio_bytes


async def synthesize_reply_audio(text):
    """Turn a text reply into a WhatsApp-ready voice note.
    Returns (audio_bytes, mime_type, filename)."""
    text = clean_text_for_speech(text)

    if TTS_ENGINE == "gemini":
        try:
            pcm = await asyncio.to_thread(gemini_text_to_speech, text)
            audio = await asyncio.to_thread(encode_pcm_for_whatsapp, pcm)
            print("Voice reply generated with Gemini TTS")
            return audio
        except Exception as e:
            print(f"Gemini TTS failed, falling back to edge-tts: {e}")

    audio_bytes = await edge_text_to_speech(text)
    print("Voice reply generated with edge-tts")
    return audio_bytes, "audio/mpeg", "reply.mp3"


def send_whatsapp_voice(to_number, audio_bytes, mime_type="audio/mpeg", filename="reply.mp3"):
    """Upload audio to WhatsApp's media store, then send it as a voice note.
    Returns True if it was sent, False otherwise."""
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}

    upload_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/media"
    files = {"file": (filename, audio_bytes, mime_type)}
    data = {"messaging_product": "whatsapp", "type": mime_type}
    upload_response = requests.post(
        upload_url, headers=headers, files=files, data=data, timeout=30
    ).json()
    media_id = upload_response.get("id")
    if not media_id:
        print("Voice upload failed:", upload_response)
        return False

    send_url = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "audio",
        "audio": {"id": media_id},
    }
    response = requests.post(
        send_url,
        headers={**headers, "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    if response.status_code != 200:
        print("WhatsApp voice send failed:", response.status_code, response.text)
        return False
    return True


# ---------- FastAPI app: hosts the webhook AND the Gradio UI together ----------
app = FastAPI()

# Track WhatsApp message IDs we've already replied to, so Meta's automatic
# retries (when our server is slow to respond) don't trigger duplicate replies.
processed_message_ids = set()


# ====================== Google Sheets ======================
GOOGLE_SHEETS_CREDENTIALS_JSON = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON")
# Paste your Google Sheet link (or just its ID) between the quotes to use it directly from the code.
# If this is filled in, it wins over the Railway variable GOOGLE_SHEET_ID.
GOOGLE_SHEET_ID_OVERRIDE = "https://docs.google.com/spreadsheets/d/1XOIWcVnmmc7XQLN1lL27B0zBWQYEQX30whD7R0_Mn8E/edit?usp=sharing"
GOOGLE_SHEET_ID = (GOOGLE_SHEET_ID_OVERRIDE or os.environ.get("GOOGLE_SHEET_ID") or "").strip()
_m = re.search(r"/d/([A-Za-z0-9_-]+)", GOOGLE_SHEET_ID)  # accept a pasted full sheet URL too
if _m:
    GOOGLE_SHEET_ID = _m.group(1)
GOOGLE_SHEET_ID = GOOGLE_SHEET_ID or None


def _friendly_sheet_error(e):
    msg = f"{type(e).__name__}: {e}"
    if "<!DOCTYPE" in msg or "<html" in msg.lower():
        return ("Google sent back a web page instead of sheet data. Most likely GOOGLE_SHEET_ID is wrong "
                "(it must be only the code between /d/ and /edit in the sheet link) or the sheet is not a normal "
                f"Google Sheet / not shared with the service account. Current ID length: {len(GOOGLE_SHEET_ID or '')} (normally 44).")
    return msg[:300]
STUDENTS_TAB = os.environ.get("GOOGLE_SHEET_TAB_NAME", "Student Leads")
AGENT_NAME = os.environ.get("SHEET_AGENT_NAME", "AI Agent")
SUMMARY_INTERVAL_SECONDS = 2 * 60 * 60
# Same layout as the team's daily sheet: A agent, B contact, C name, D remarks, E course (dropdown), F QA remark
STUDENT_HEADERS = ["Agent name", "Customer contact #", "Customer Name", "Remarks", "interested course", "QA Remark"]
SUMMARY_HEADERS = ["Time window", "Total numbers", "New", "Returning", "Interested", "Not Interested",
                   "Fee asked", "Discount requests", "Forwarded to head", "Voice notes received", "Media sent"]
DETAIL_HEADERS = ["Window end", "Phone", "Name", "Student asked", "Agent replied", "Status"]
def _norm_phone(v):
    d = re.sub(r"\D", "", str(v))
    return d[-10:] if len(d) >= 10 else d


_ss = _students = _summary = _details = None
_row_of = {}
_manual_rows = set()
_sheet_state = {"connected": False, "error": "", "last_ok": "", "last_error": "", "tab": "", "email": "", "syncing": ""}
_sheet_lock = threading.Lock()

if GOOGLE_SHEETS_CREDENTIALS_JSON and GOOGLE_SHEET_ID:
    try:
        import gspread
        from google.oauth2.service_account import Credentials as GoogleCredentials
        try:
            _sheet_state["email"] = json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON).get("client_email", "")
        except Exception:
            pass

        _gc = gspread.authorize(GoogleCredentials.from_service_account_info(
            json.loads(GOOGLE_SHEETS_CREDENTIALS_JSON),
            scopes=["https://www.googleapis.com/auth/spreadsheets"]))
        _ss = _gc.open_by_key(GOOGLE_SHEET_ID)

        def _tab(name, headers):
            try:
                return _ss.worksheet(name)
            except gspread.WorksheetNotFound:
                ws = _ss.add_worksheet(name, rows=2000, cols=len(headers))
                ws.append_row(headers)
                return ws

        _students_existed = STUDENTS_TAB in [w.title for w in _ss.worksheets()]
        _students = _tab(STUDENTS_TAB, STUDENT_HEADERS)
        _summary = _tab("2 Hour Summary", SUMMARY_HEADERS)
        _details = _tab("2 Hour Details", DETAIL_HEADERS)
        if not _students_existed:
            try:  # header look + frozen row + course dropdown (column E) for a brand-new tab
                _reqs = [
                    {"repeatCell": {"range": {"sheetId": _students.id, "startRowIndex": 0, "endRowIndex": 1},
                                    "cell": {"userEnteredFormat": {"textFormat": {"bold": True},
                                             "backgroundColor": {"red": 0.6, "green": 0.6, "blue": 0.6}}},
                                    "fields": "userEnteredFormat(textFormat,backgroundColor)"}},
                    {"updateSheetProperties": {"properties": {"sheetId": _students.id,
                                               "gridProperties": {"frozenRowCount": 1}},
                                               "fields": "gridProperties.frozenRowCount"}}]
                if ALL_COURSE_NAMES:
                    _reqs.append({"setDataValidation": {
                        "range": {"sheetId": _students.id, "startRowIndex": 1, "endRowIndex": 2000,
                                  "startColumnIndex": 4, "endColumnIndex": 5},
                        "rule": {"condition": {"type": "ONE_OF_LIST",
                                               "values": [{"userEnteredValue": c} for c in ALL_COURSE_NAMES]},
                                 "showCustomUi": True, "strict": False}}})
                _ss.batch_update({"requests": _reqs})
            except Exception as e:
                print(f"Students tab styling skipped: {e}")
        _colA = _students.col_values(1)
        _row_of = {_norm_phone(v): i + 1 for i, v in enumerate(_students.col_values(2)) if i > 0 and v}
        _manual_rows = {i + 1 for i, v in enumerate(_colA) if i > 0 and v.strip() and v.strip() != AGENT_NAME}
        _sheet_state.update(connected=True, tab=STUDENTS_TAB, error="")
        print("Google Sheets connected.")
    except Exception as e:
        _sheet_state["error"] = _friendly_sheet_error(e)
        print(f"Google Sheets setup failed (sheets skipped): {e}")
else:
    _missing = [n for n, v in (("GOOGLE_SHEETS_CREDENTIALS_JSON", GOOGLE_SHEETS_CREDENTIALS_JSON), ("GOOGLE_SHEET_ID", GOOGLE_SHEET_ID)) if not v]
    _sheet_state["error"] = "Railway variable(s) missing: " + ", ".join(_missing)
    print("Google Sheets env vars not set - sheets disabled.")


def _match_course(c):
    """Map the AI's course text to the exact dropdown name when possible."""
    n = lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower())
    if not c:
        return ""
    for name in ALL_COURSE_NAMES:
        if n(c) == n(name) or (n(c) and (n(c) in n(name) or n(name) in n(c))):
            return name
    return c


def sync_student_row(phone):
    """One row per student, updated live. Never touches the 'QA Remark' column (F).
    Rows typed by hand (agent name other than AGENT_NAME) are only topped up, never overwritten."""
    if not _students:
        return
    try:
        p = get_profile(phone)
        key = _norm_phone(phone)
        course = _match_course(p["course"])
        with _sheet_lock:
            r = _row_of.get(key)
            if not r:
                res = _students.append_row([AGENT_NAME, phone, p["name"], p["remarks"], course, ""],
                                           value_input_option="RAW")
                m = re.search(r"!A(\d+)", res["updates"]["updatedRange"])
                r = int(m.group(1))
                _row_of[key] = r
            elif r in _manual_rows:
                cur = _students.row_values(r)
                cur += [""] * (6 - len(cur))
                upd = []
                if p["name"] and not cur[2].strip():
                    upd.append({"range": f"C{r}", "values": [[p["name"]]]})
                if course and not cur[4].strip():
                    upd.append({"range": f"E{r}", "values": [[course]]})
                last = (p["remarks"] or "").split(" | ")[-1].strip()
                if last and last not in cur[3]:
                    upd.append({"range": f"D{r}", "values": [[(cur[3] + " | " if cur[3].strip() else "") + last]]})
                if upd:
                    _students.batch_update(upd, value_input_option="RAW")
            else:
                _students.batch_update([{"range": f"B{r}:E{r}", "values": [[phone, p["name"], p["remarks"], course]]}],
                                       value_input_option="RAW")
            yellow = {"backgroundColor": {"red": 1, "green": 1, "blue": 0}}
            white = {"backgroundColor": {"red": 1, "green": 1, "blue": 1}}
            _students.format(f"A{r}:F{r}", yellow if p["status"] == "Not Interested" else white)
        _sheet_state["last_ok"] = now()
    except Exception as e:
        _sheet_state["last_error"] = f"{now()} UTC - {_friendly_sheet_error(e)}"[:400]
        print(f"Student sheet sync failed for {phone}: {e}")


def extract_and_sync(phone):
    """Real-time: read the chat, update the student's profile + sheet row."""
    try:
        p = get_profile(phone)
        prompt = f"""Read this WhatsApp chat between a potential student and Leaders Academia's assistant.
Return STRICT JSON only, no other text:
{{"name":"","city":"","qualification":"","motive":"","course":"","status":"Interested | Not Interested | Follow-up Needed | Enrolled (pick one)","discount_requested":false,"no_voice":false,"remark":"max 12 words: what the STUDENT said or asked in the latest messages, starting with Student: (example: Student: fee poochi, AI course mein interest)"}}
Use "" when unknown. no_voice=true only if the student asked not to get voice messages. Already known: {profile_text(p)}

CHAT:
{format_history(get_history(phone))}"""
        raw = generate_text(prompt)
        d = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        upd = {k: d[k].strip() for k in ("name", "city", "qualification", "motive", "course")
               if isinstance(d.get(k), str) and d[k].strip()
               and d[k].strip().lower() not in ("not provided", "unknown", "none", "n/a")}
        if d.get("status") in STATUSES:
            upd["status"] = d["status"]
        if d.get("discount_requested"):
            upd["discount"] = 1
        if d.get("no_voice"):
            upd["no_voice"] = 1
        rem = (d.get("remark") or "").strip()
        if rem and rem not in p["remarks"]:
            upd["remarks"] = (p["remarks"] + " | " if p["remarks"] else "") + datetime.datetime.now().strftime("%d %b") + " " + rem
        update_profile(phone, **upd)
        sync_student_row(phone)
    except Exception as e:
        print(f"Profile extraction failed for {phone}: {e}")
        sync_student_row(phone)


def write_summary(start, end):
    if not _summary:
        return
    ph = [r["phone"] for r in db("SELECT DISTINCT phone FROM messages WHERE direction='in' AND time>?", (start,))]
    if not ph:
        return
    first = {r["phone"] for r in db("SELECT phone FROM messages GROUP BY phone HAVING MIN(time)>?", (start,))}
    st = {s: 0 for s in STATUSES}
    disc = 0
    for x in ph:
        p = get_profile(x)
        st[p["status"]] = st.get(p["status"], 0) + 1
        disc += 1 if p["discount"] else 0
    fee = db("""SELECT COUNT(DISTINCT phone) c FROM messages WHERE direction='in' AND time>? AND
      (lower(text) LIKE '%fee%' OR lower(text) LIKE '%price%' OR lower(text) LIKE '%kitn%' OR lower(text) LIKE '%cost%')""", (start,))[0]["c"]
    voice = db("SELECT COUNT(*) c FROM messages WHERE direction='in' AND type='voice' AND time>?", (start,))[0]["c"]
    fwd = db("SELECT COUNT(*) c FROM profiles WHERE fwd_time>?", (start,))[0]["c"]
    med = db("SELECT COUNT(*) c FROM media_sent WHERE time>?", (start,))[0]["c"]
    new_n = len(first & set(ph))
    _summary.append_row([f"{start} to {end}", len(ph), new_n, len(ph) - new_n, st["Interested"],
                         st["Not Interested"], fee, disc, fwd, voice, med], value_input_option="RAW")
    rows = []
    for x in ph:
        p = get_profile(x)
        q = db("SELECT text FROM messages WHERE phone=? AND direction='in' AND time>? ORDER BY id DESC LIMIT 1", (x, start))
        a = db("SELECT text FROM messages WHERE phone=? AND direction='out' AND time>? ORDER BY id DESC LIMIT 1", (x, start))
        rows.append([end, x, p["name"], q[0]["text"][:300] if q else "", a[0]["text"][:300] if a else "", p["status"]])
    _details.append_rows(rows, value_input_option="RAW")


async def run_periodic_summary():
    last = now()
    while True:
        await asyncio.sleep(SUMMARY_INTERVAL_SECONDS)
        start, last = last, now()
        try:
            await asyncio.to_thread(write_summary, start, last)
        except Exception as e:
            print(f"2-hour summary failed: {e}")


# ====================== WhatsApp helpers: media, templates ======================
ADMIN_ALERT_NUMBER = os.environ.get("ADMIN_ALERT_NUMBER")          # gets a WhatsApp alert on every forward
WA_FOLLOWUP_TEMPLATE = os.environ.get("WA_FOLLOWUP_TEMPLATE")      # approved Meta template (1 variable: name)
WA_CHECKIN_TEMPLATE = os.environ.get("WA_CHECKIN_TEMPLATE")        # approved Meta template (1 variable: name)
WA_TEMPLATE_LANG = os.environ.get("WA_TEMPLATE_LANG", "en")
WA_URL = f"https://graph.facebook.com/v25.0/{WHATSAPP_PHONE_NUMBER_ID}"


def send_whatsapp_image(to_number, data, mime, caption):
    h = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    up = requests.post(f"{WA_URL}/media", headers=h, timeout=30,
                       files={"file": ("poster." + mime.split("/")[-1], data, mime)},
                       data={"messaging_product": "whatsapp", "type": mime}).json()
    if not up.get("id"):
        print("Image upload failed:", up)
        return False
    r = requests.post(f"{WA_URL}/messages", headers={**h, "Content-Type": "application/json"}, timeout=20,
                      json={"messaging_product": "whatsapp", "to": to_number, "type": "image",
                            "image": {"id": up["id"], "caption": caption}})
    return r.status_code == 200


def pick_media(phone, text):
    t = text.lower()
    for m in db("SELECT id,title,keywords,mime FROM media"):
        if any(k.strip() and k.strip().lower() in t for k in m["keywords"].split(",")):
            if not db("SELECT 1 FROM media_sent WHERE phone=? AND media_id=?", (phone, m["id"])):
                return m
    return None


def send_template_or_text(to, template, name, text):
    """Outside WhatsApp's 24-hour window only approved templates are delivered."""
    if template:
        r = requests.post(f"{WA_URL}/messages", timeout=20,
                          headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}", "Content-Type": "application/json"},
                          json={"messaging_product": "whatsapp", "to": to, "type": "template",
                                "template": {"name": template, "language": {"code": WA_TEMPLATE_LANG},
                                             "components": [{"type": "body", "parameters": [{"type": "text", "text": name or "there"}]}]}})
    else:
        r = send_whatsapp_message(to, text)
    if r.status_code != 200:
        print("Follow-up send failed:", r.status_code, r.text)
    return r.status_code == 200


FOLLOWUP_AFTER_HOURS = float(os.environ.get("FOLLOWUP_AFTER_HOURS", "20"))   # no reply from an interested/undecided student
CHECKIN_AFTER_HOURS = float(os.environ.get("CHECKIN_AFTER_HOURS", "20"))     # after the student was sent to the team head
FOLLOWUP_START_HOUR = int(os.environ.get("FOLLOWUP_START_HOUR", "10"))       # Pakistan time window for sending
FOLLOWUP_END_HOUR = int(os.environ.get("FOLLOWUP_END_HOUR", "20"))


def _hours_ago(h):
    return (datetime.datetime.now() - datetime.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S")


def _can_send_followup(p, template):
    """Without an approved template WhatsApp only delivers free text within 24h of the student's last message."""
    if template:
        return True
    try:
        last = datetime.datetime.strptime(p["last_in"], "%Y-%m-%d %H:%M:%S")
        return (datetime.datetime.now() - last).total_seconds() < 23.5 * 3600
    except Exception:
        return False


def do_followups():
    if not (FOLLOWUP_START_HOUR <= pk_now().hour < FOLLOWUP_END_HOUR):
        return  # do not message students late at night
    rows = db("""SELECT * FROM profiles WHERE status IN ('Interested','Follow-up Needed') AND followup_sent=0
                 AND fwd_count=0 AND bot_paused=0 AND last_in!='' AND last_in<?
                 AND (SELECT COUNT(*) FROM messages m WHERE m.phone=profiles.phone AND m.direction='in')>=2""",
              (_hours_ago(FOLLOWUP_AFTER_HOURS),))
    for p in rows:
        if not _can_send_followup(p, WA_FOLLOWUP_TEMPLATE):
            continue
        n = p["name"] or "there"
        about = f"{p['course']} course" if p["course"] else "Leaders Academia ke courses"
        text = f"Assalamualaikum {n}! Aap ne {about} ke baare mein baat ki thi. Koi sawal ho to bata dein, main madad kar dungi."
        if send_template_or_text(p["phone"], WA_FOLLOWUP_TEMPLATE, p["name"], text):
            log_message(p["phone"], "out", "text", text)
            update_profile(p["phone"], followup_sent=1, remarks=(p["remarks"] + " | " if p["remarks"] else "") + datetime.datetime.now().strftime("%d %b") + " follow-up sent")
            sync_student_row(p["phone"])
    for p in db("SELECT * FROM profiles WHERE fwd_count>0 AND checkin_sent=0 AND bot_paused=0 AND fwd_time!='' AND fwd_time<?", (_hours_ago(CHECKIN_AFTER_HOURS),)):
        if not _can_send_followup(p, WA_CHECKIN_TEMPLATE):
            continue
        n = p["name"] or "there"
        text = f"Assalamualaikum {n}! Kya aap ki {TEAM_HEAD_NAME} se baat ho gayi? Unhon ne kya kaha, bata dijiye."
        if send_template_or_text(p["phone"], WA_CHECKIN_TEMPLATE, p["name"], text):
            log_message(p["phone"], "out", "text", text)
            update_profile(p["phone"], checkin_sent=1, remarks=(p["remarks"] + " | " if p["remarks"] else "") + datetime.datetime.now().strftime("%d %b") + " head check-in sent")
            sync_student_row(p["phone"])


async def run_followups():
    while True:
        await asyncio.sleep(900)
        try:
            await asyncio.to_thread(do_followups)
        except Exception as e:
            print(f"Follow-up job failed: {e}")


@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(run_periodic_summary())
    asyncio.create_task(run_followups())


# ====================== WhatsApp webhook ======================
@app.get("/webhook")
def verify_webhook(request: Request):
    q = request.query_params
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(q.get("hub.challenge"))
    return PlainTextResponse("Verification failed", status_code=403)


async def handle_user_text(frm, user_text, is_voice):
    log_message(frm, "in", "voice" if is_voice else "text", user_text)
    p = get_profile(frm)
    if p["bot_paused"]:  # admin has taken over this chat: log it, update the sheet, but do not reply
        asyncio.create_task(asyncio.to_thread(extract_and_sync, frm))
        if ADMIN_ALERT_NUMBER:
            await asyncio.to_thread(send_whatsapp_message, ADMIN_ALERT_NUMBER,
                                    f"Student replied in a PAUSED chat: {p['name'] or 'name unknown'} ({frm}): {user_text[:150]}")
        return
    reply = await asyncio.to_thread(rag_answer, user_text, frm, bool(is_voice and not p["no_voice"]))
    forwarded = record_forward_if_needed(frm, reply)
    head = f"{TEAM_HEAD_NAME}: {TEAM_HEAD_NUMBER}"
    text_reply = reply.replace("[[HEAD]]", head).strip()
    spoken = reply.replace("[[HEAD]]", "").strip()
    use_voice = (not p["no_voice"]) and (is_voice or len(spoken) > 450)  # long answers go out as a voice note
    log_message(frm, "out", "voice" if use_voice else "text", text_reply)
    sent = False
    if use_voice:
        try:
            audio, mime, fn = await synthesize_reply_audio(spoken)
            sent = await asyncio.to_thread(send_whatsapp_voice, frm, audio, mime, fn)
        except Exception as e:
            print(f"Voice reply failed: {e}")
        if sent and forwarded:  # number is never spoken, only typed
            await asyncio.to_thread(send_whatsapp_message, frm, head)
    if not sent:
        await asyncio.to_thread(send_whatsapp_message, frm, text_reply)
    m = await asyncio.to_thread(pick_media, frm, user_text)
    if m:
        row = db("SELECT data FROM media WHERE id=?", (m["id"],))[0]
        if await asyncio.to_thread(send_whatsapp_image, frm, row["data"], m["mime"], m["title"]):
            db("INSERT INTO media_sent VALUES(?,?,?)", (frm, m["id"], now()), write=True)
    if forwarded and ADMIN_ALERT_NUMBER:
        p2 = get_profile(frm)
        await asyncio.to_thread(send_whatsapp_message, ADMIN_ALERT_NUMBER,
                                f"Student forwarded: {p2['name'] or 'name unknown'} ({frm}) - {p2['course'] or 'course not set'}. Last message: {user_text[:150]}")
    asyncio.create_task(asyncio.to_thread(extract_and_sync, frm))


@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    data = await request.json()
    try:
        messages = data["entry"][0]["changes"][0]["value"].get("messages")
        if messages:
            message = messages[0]
            mid, frm, mtype = message.get("id"), message["from"], message.get("type")
            if mid in processed_message_ids:
                return {"status": "duplicate, skipped"}
            processed_message_ids.add(mid)
            user_text, is_voice = None, False
            if mtype == "text":
                user_text = message.get("text", {}).get("body", "")
            elif mtype == "audio":
                is_voice = True
                try:
                    audio_bytes, mime = await asyncio.to_thread(download_whatsapp_media, message["audio"]["id"])
                    user_text = await asyncio.to_thread(transcribe_audio, audio_bytes, mime)
                except Exception as e:
                    print(f"Voice transcription failed: {e}")
                    await asyncio.to_thread(send_whatsapp_message, frm,
                                            "Yeh voice note samajhne mein masla ho raha hai. Dobara bhej dein ya text mein likh dein.")
            if user_text:
                await handle_user_text(frm, user_text, is_voice)
    except Exception as e:
        print("Error processing incoming WhatsApp message:", e)
    return {"status": "received"}


# ====================== Dashboard ======================
from fastapi.responses import JSONResponse, Response

DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "changeme")


def _token():
    return hmac.new(DASHBOARD_PASSWORD.encode(), b"leaders-dash", hashlib.sha256).hexdigest()


def check_auth(request: Request):
    if not hmac.compare_digest(request.cookies.get("dash", ""), _token()):
        raise HTTPException(status_code=401, detail="login required")
    return True


@app.post("/dashboard/login")
async def dash_login(request: Request):
    d = await request.json()
    if d.get("user") == DASHBOARD_USER and hmac.compare_digest(str(d.get("password", "")), DASHBOARD_PASSWORD):
        r = JSONResponse({"ok": True})
        r.set_cookie("dash", _token(), httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
        return r
    return JSONResponse({"ok": False}, status_code=401)


@app.post("/dashboard/logout")
async def dash_logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie("dash")
    return r


@app.get("/dashboard/api/chats")
def dash_chats(filter: str = "all", hours: int = 0, auth: bool = Depends(check_auth)):
    since = (datetime.datetime.now() - datetime.timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S") if hours else ""
    rows = db("""SELECT * FROM (SELECT p.*,
      (SELECT time FROM messages m WHERE m.phone=p.phone ORDER BY id DESC LIMIT 1) AS last_time,
      (SELECT text FROM messages m WHERE m.phone=p.phone ORDER BY id DESC LIMIT 1) AS last_text FROM profiles p)
      WHERE last_time IS NOT NULL AND (?='all' OR status=?) AND last_time>=? ORDER BY last_time DESC LIMIT 300""",
              (filter, filter, since))
    return {"chats": rows}


@app.get("/dashboard/api/chat/{phone}")
def dash_chat(phone: str, auth: bool = Depends(check_auth)):
    return {"messages": db("SELECT id,time,direction,type,text,good,by_admin FROM messages WHERE phone=? ORDER BY id", (phone,))}


@app.post("/dashboard/api/pause/{phone}")
async def dash_pause(phone: str, request: Request, auth: bool = Depends(check_auth)):
    """Pause / resume the agent for one chat (admin takes over)."""
    body = await request.json()
    get_profile(phone)
    update_profile(phone, bot_paused=1 if body.get("paused") else 0)
    return {"ok": True, "paused": bool(body.get("paused"))}


@app.post("/dashboard/api/send")
async def dash_send(request: Request, auth: bool = Depends(check_auth)):
    """Admin writes to a student from the dashboard (shown in the chat as an Admin message)."""
    body = await request.json()
    phone = str(body.get("phone", "")).strip()
    text = str(body.get("text", "")).strip()
    if not phone or not text:
        return JSONResponse({"ok": False, "error": "Write a message first"}, status_code=400)
    if len(text) > 3500:
        return JSONResponse({"ok": False, "error": "Message is too long"}, status_code=400)
    try:
        r = await asyncio.to_thread(send_whatsapp_message, phone, text)
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Could not reach WhatsApp: {e}"}, status_code=502)
    if r.status_code != 200:
        try:
            detail = r.json().get("error", {}).get("message", "")
        except Exception:
            detail = ""
        hint = " WhatsApp only allows free messages within 24 hours of the student's last message." if r.status_code in (400, 403) else ""
        return JSONResponse({"ok": False, "error": (detail or f"WhatsApp error {r.status_code}") + hint}, status_code=502)
    db("INSERT INTO messages(phone,time,direction,type,text,by_admin) VALUES(?,?,?,?,?,1)", (phone, now(), "out", "text", text), write=True)
    return {"ok": True}


@app.post("/dashboard/api/good/{mid}")
def dash_good(mid: int, auth: bool = Depends(check_auth)):
    db("UPDATE messages SET good=1-good WHERE id=?", (mid,), write=True)
    return {"ok": True}


@app.get("/dashboard/api/people")
def dash_people(kind: str, auth: bool = Depends(check_auth)):
    if kind == "fol":
        rows = db("""SELECT phone,name,course,last_in,followup_sent FROM profiles
                     WHERE status IN ('Interested','Follow-up Needed') AND last_in!='' AND fwd_count=0
                     AND (SELECT COUNT(*) FROM messages m WHERE m.phone=profiles.phone AND m.direction='in')>=2
                     ORDER BY last_in DESC""")
        due = _hours_ago(FOLLOWUP_AFTER_HOURS)
        for r in rows:
            r["state"] = "Follow-up sent" if r["followup_sent"] else ("Due now" if r["last_in"] < due else f"Waiting (under {FOLLOWUP_AFTER_HOURS:g}h)")
    else:
        rows = db("SELECT phone,name,course,fwd_count,fwd_time,checkin_sent FROM profiles WHERE fwd_count>0 ORDER BY fwd_time DESC")
        for r in rows:
            r["state"] = "Check-in sent" if r["checkin_sent"] else f"Waiting for {CHECKIN_AFTER_HOURS:g}h check-in"
    return {"people": rows}


@app.get("/dashboard/api/media")
def dash_media_list(auth: bool = Depends(check_auth)):
    return {"media": db("SELECT id,title,keywords FROM media ORDER BY id DESC")}


@app.post("/dashboard/api/media")
async def dash_media_add(request: Request, title: str = "", keywords: str = "", auth: bool = Depends(check_auth)):
    mime = request.headers.get("content-type", "image/jpeg")
    db("INSERT INTO media(title,keywords,mime,data) VALUES(?,?,?,?)", (title, keywords, mime, await request.body()), write=True)
    return {"ok": True}


@app.delete("/dashboard/api/media/{mid}")
def dash_media_del(mid: int, auth: bool = Depends(check_auth)):
    db("DELETE FROM media WHERE id=?", (mid,), write=True)
    return {"ok": True}


@app.get("/dashboard/media/{mid}")
def dash_media_file(mid: int, auth: bool = Depends(check_auth)):
    r = db("SELECT mime,data FROM media WHERE id=?", (mid,))
    return Response(r[0]["data"], media_type=r[0]["mime"]) if r else Response(status_code=404)


@app.get("/dashboard/api/analytics")
def dash_analytics(days: int = 14, tz: int = 0, auth: bool = Depends(check_auth)):
    """Daily numbers for the Analytics page. tz = browser offset from UTC in minutes (server time is stored as UTC)."""
    days = max(1, min(days, 90))
    tz = max(-840, min(tz, 840))
    off = f"{tz:+d} minutes"
    today = (datetime.datetime.now() + datetime.timedelta(minutes=tz)).date()
    labels = [(today - datetime.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    since = (datetime.datetime.now() - datetime.timedelta(days=days + 1)).strftime("%Y-%m-%d %H:%M:%S")

    def per_day(rows):
        m = {r["d"]: r["c"] for r in rows}
        return [m.get(d, 0) for d in labels]

    msg_in = per_day(db("SELECT date(time,?) d, COUNT(*) c FROM messages WHERE direction='in' AND time>=? GROUP BY d", (off, since)))
    msg_out = per_day(db("SELECT date(time,?) d, COUNT(*) c FROM messages WHERE direction='out' AND time>=? GROUP BY d", (off, since)))
    voice = per_day(db("SELECT date(time,?) d, COUNT(*) c FROM messages WHERE direction='in' AND type='voice' AND time>=? GROUP BY d", (off, since)))
    active = per_day(db("SELECT date(time,?) d, COUNT(DISTINCT phone) c FROM messages WHERE direction='in' AND time>=? GROUP BY d", (off, since)))
    fwd = per_day(db("SELECT date(fwd_time,?) d, COUNT(*) c FROM profiles WHERE fwd_time!='' AND fwd_time>=? GROUP BY d", (off, since)))
    media = per_day(db("SELECT date(time,?) d, COUNT(*) c FROM media_sent WHERE time>=? GROUP BY d", (off, since)))
    new_by_status = {st: [0] * len(labels) for st in STATUSES}
    for r in db("""SELECT date(f,?) d, status, COUNT(*) c FROM
                   (SELECT MIN(m.time) f, p.status status FROM messages m JOIN profiles p ON p.phone=m.phone GROUP BY m.phone)
                   GROUP BY d, status""", (off,)):
        if r["d"] in labels and r["status"] in new_by_status:
            new_by_status[r["status"]][labels.index(r["d"])] = r["c"]
    hours = [0] * 24
    for r in db("SELECT CAST(strftime('%H',time,?) AS INTEGER) h, COUNT(*) c FROM messages WHERE direction='in' AND time>=? GROUP BY h", (off, since)):
        hours[r["h"]] = r["c"]
    status = {r["status"]: r["c"] for r in db(
        "SELECT status, COUNT(*) c FROM profiles WHERE phone IN (SELECT phone FROM messages) GROUP BY status")}
    courses = [[r["n"], r["c"]] for r in db(
        """SELECT MIN(course) n, COUNT(*) c FROM profiles WHERE course!='' AND phone IN (SELECT phone FROM messages)
           GROUP BY lower(trim(course)) ORDER BY c DESC LIMIT 8""")]
    cities = [[r["n"], r["c"]] for r in db(
        """SELECT MIN(city) n, COUNT(*) c FROM profiles WHERE city!='' AND phone IN (SELECT phone FROM messages)
           GROUP BY lower(trim(city)) ORDER BY c DESC LIMIT 6""")]
    total_students = sum(status.values())
    return {"labels": labels, "msg_in": msg_in, "msg_out": msg_out, "voice": voice, "active": active,
            "fwd": fwd, "media": media, "new_by_status": new_by_status, "hours": hours, "status": status,
            "courses": courses, "cities": cities,
            "totals": {"students": total_students,
                       "messages": db("SELECT COUNT(*) c FROM messages")[0]["c"],
                       "forwarded": db("SELECT COUNT(*) c FROM profiles WHERE fwd_count>0")[0]["c"],
                       "discount": db("SELECT COUNT(*) c FROM profiles WHERE discount=1")[0]["c"]}}


def _sheet_wipe_all():
    """Remove all data rows (keeps header row) from the three bot tabs."""
    if not _ss:
        return
    for ws in (_students, _summary, _details):
        if ws:
            try:
                n = max(ws.row_count, 2)
                ws.batch_clear([f"A2:Z{n}"])
                ws.format(f"A2:Z{n}", {"backgroundColor": {"red": 1, "green": 1, "blue": 1}})
            except Exception as e:
                print(f"Sheet clear failed for {getattr(ws, 'title', '?')}: {e}")
    _row_of.clear()
    _manual_rows.clear()


@app.post("/dashboard/api/clear-all")
async def dash_clear_all(request: Request, auth: bool = Depends(check_auth)):
    """Delete ALL chat history, student profiles and sheet rows. Posters (media) are kept."""
    body = await request.json()
    if body.get("confirm") != "DELETE":
        return JSONResponse({"ok": False, "error": "Type DELETE to confirm"}, status_code=400)
    for t in ("messages", "profiles", "media_sent"):
        db(f"DELETE FROM {t}", write=True)
    with _sheet_lock:
        await asyncio.to_thread(_sheet_wipe_all)
    return {"ok": True}


@app.delete("/dashboard/api/chat/{phone}")
def dash_delete_chat(phone: str, auth: bool = Depends(check_auth)):
    """Delete one student's chat, profile and sheet row."""
    for t in ("messages", "profiles", "media_sent"):
        db(f"DELETE FROM {t} WHERE phone=?", (phone,), write=True)
    key = _norm_phone(phone)
    with _sheet_lock:
        r = _row_of.pop(key, None)
        try:
            if r and _students:
                _students.delete_rows(r)
                for k, v in list(_row_of.items()):
                    if v > r:
                        _row_of[k] = v - 1
                _manual_rows.discard(r)
                for x in {m for m in _manual_rows if m > r}:
                    _manual_rows.discard(x)
                    _manual_rows.add(x - 1)
            if _details:
                vals = _details.col_values(2)
                for i in range(len(vals), 1, -1):
                    if _norm_phone(vals[i - 1]) == key:
                        _details.delete_rows(i)
        except Exception as e:
            print(f"Sheet row delete failed for {phone}: {e}")
    return {"ok": True}


@app.get("/dashboard/api/ai")
def dash_ai_status(auth: bool = Depends(check_auth)):
    return {**_ai_state, "groq": bool(groq_client)}


@app.get("/dashboard/api/sheet")
def dash_sheet_status(auth: bool = Depends(check_auth)):
    return {**_sheet_state, "rows": len(_row_of)}


@app.post("/dashboard/api/sheet/sync")
def dash_sheet_sync(auth: bool = Depends(check_auth)):
    """Push every known student to the sheet (also fills in students who chatted before the sheet worked)."""
    if not _students:
        return {"ok": False, "error": _sheet_state["error"] or "Google Sheets is not connected"}
    if _sheet_state["syncing"]:
        return {"ok": True, "started": False}
    phones = [r["phone"] for r in db("SELECT DISTINCT phone FROM messages")]

    def run():
        _sheet_state["syncing"] = f"0/{len(phones)}"
        for i, ph in enumerate(phones, 1):
            sync_student_row(ph)
            _sheet_state["syncing"] = f"{i}/{len(phones)}"
            time.sleep(1.5)  # stay under Google's per-minute write quota
        _sheet_state["syncing"] = ""

    threading.Thread(target=run, daemon=True).start()
    return {"ok": True, "started": True, "count": len(phones)}


DASHBOARD_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Leaders Academia Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Arial,sans-serif;background:#f4f5f7;color:#1a1a1a}
#login{height:100vh;display:flex;align-items:center;justify-content:center}
#login form{background:#fff;padding:28px;border-radius:12px;width:300px;box-shadow:0 2px 12px #0002}
#login input,.f input{width:100%;padding:10px;margin:6px 0;border:1px solid #d1d5db;border-radius:8px}
button{cursor:pointer;border:0;border-radius:8px;padding:10px 12px;background:#2563eb;color:#fff}
#app{display:none;height:100vh}nav{width:190px;background:#111827;padding:12px;display:flex;flex-direction:column;gap:6px}
nav button{background:transparent;text-align:left;color:#d1d5db}nav button.on{background:#2563eb;color:#fff}
main{flex:1;display:flex;min-width:0}#list{width:290px;overflow-y:auto;background:#fff;border-right:1px solid #e5e7eb;display:none}
#pane{flex:1;overflow-y:auto;padding:16px}.c{padding:10px 14px;border-bottom:1px solid #f0f0f0;cursor:pointer}.c.on{background:#eef2ff}
.m{max-width:72%;margin:8px 0;padding:9px 13px;border-radius:10px;white-space:pre-wrap;font-size:14px}
.m.in{background:#fff;border:1px solid #e5e7eb}.m.out{background:#2563eb;color:#fff;margin-left:auto}.m small{opacity:.7;display:block}
.g{cursor:pointer;float:right;margin-left:8px}table{width:100%;border-collapse:collapse;background:#fff}
td,th{padding:9px;border-bottom:1px solid #eee;text-align:left;font-size:14px}tr.cl{cursor:pointer}.e{color:#9ca3af;text-align:center;margin-top:30px}
.grid{display:flex;flex-wrap:wrap;gap:12px}.card{background:#fff;padding:10px;border-radius:10px;width:200px}.card img{width:100%;border-radius:6px}
</style></head><body>
<div id="login"><form onsubmit="login();return false"><h3>Leaders Academia</h3><input id="u" placeholder="Username">
<input id="p" type="password" placeholder="Password"><button style="width:100%">Sign in</button><p id="err" style="color:#b91c1c"></p></form></div>
<div id="app"><nav id="nav"></nav><main><div id="list"></div><div id="pane"></div></main></div>
<script>
const $=s=>document.querySelector(s),esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const VIEWS=[['live','💬 Live Chats'],['int','⭐ Interested'],['not','🚫 Not Interested'],['media','🖼 Media'],['hist','📜 History'],['fol','🔔 Follow-ups'],['fwd','➡️ Forwarded']];
const FILTER={live:'all',int:'Interested',not:'Not Interested',hist:'all'};let view='live',sel=null,lastSel=null;
async function api(u,o){const r=await fetch(u,o);if(r.status==401){showLogin();throw new Error('auth')}return r.json()}
function showLogin(){$('#login').style.display='flex';$('#app').style.display='none'}
async function login(){const r=await fetch('/dashboard/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user:$('#u').value,password:$('#p').value})});
 if(r.ok){$('#login').style.display='none';$('#app').style.display='flex';init()}else $('#err').textContent='Wrong username or password'}
function init(){$('#nav').innerHTML=VIEWS.map(([k,l])=>`<button id="b_${k}" onclick="go('${k}')">${l}</button>`).join('')+'<button onclick="logout()">Sign out</button>';go(view)}
async function logout(){await fetch('/dashboard/logout',{method:'POST'});showLogin()}
function go(v,p){view=v;if(p)sel=p;document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.id==='b_'+v));draw()}
async function draw(){try{
 if(view in FILTER){const L=(await api(`/dashboard/api/chats?filter=${encodeURIComponent(FILTER[view])}&hours=${view==='live'?24:0}`)).chats;
  $('#list').style.display='block';
  $('#list').innerHTML=L.map(c=>`<div class="c ${c.phone===sel?'on':''}" onclick="go(view,'${esc(c.phone)}')"><b>${esc(c.name||c.phone)}</b> <small>${c.name?esc(c.phone):''}</small><br><small>${esc(c.status)}${c.fwd_count?' · forwarded':''} · ${esc(c.last_time)}</small><br><small>${esc((c.last_text||'').slice(0,55))}</small></div>`).join('')||'<p class="e">No chats</p>';
  const f=$('#pane');if(!sel){f.innerHTML='<p class="e">Select a chat</p>';return}
  const M=(await api('/dashboard/api/chat/'+sel)).messages,nb=f.scrollHeight-f.scrollTop-f.clientHeight<90;
  f.innerHTML=M.map(m=>`<div class="m ${m.direction}"><small>${m.direction==='in'?'Student':'Agent'} · ${m.type} · ${esc(m.time)}${m.direction==='out'?`<span class="g" onclick="good(${m.id})">${m.good?'👍 good example':'👍'}</span>`:''}</small>${esc(m.text)}</div>`).join('');
  if(nb||lastSel!==sel)f.scrollTop=f.scrollHeight;lastSel=sel;
 }else if(view==='fol'||view==='fwd'){$('#list').style.display='none';
  const P=(await api('/dashboard/api/people?kind='+view)).people;
  $('#pane').innerHTML='<table><tr><th>Number</th><th>Name</th><th>Course</th><th>'+(view==='fol'?'Last message':'Forwarded')+'</th><th>State</th></tr>'+P.map(r=>`<tr class="cl" onclick="go('hist','${esc(r.phone)}')"><td>${esc(r.phone)}</td><td>${esc(r.name)}</td><td>${esc(r.course)}</td><td>${esc(r.last_in||r.fwd_time)}</td><td>${esc(r.state)}</td></tr>`).join('')+'</table>'+(P.length?'':'<p class="e">Nothing here yet</p>');
 }else if(view==='media'){$('#list').style.display='none';await drawMedia()}
}catch(e){}}
async function good(id){await fetch('/dashboard/api/good/'+id,{method:'POST'});draw()}
async function drawMedia(){const M=(await api('/dashboard/api/media')).media;
 $('#pane').innerHTML=`<div class="f card" style="width:340px"><b>Add poster</b><input id="mt" placeholder="Caption / title (e.g. AI Automation Course)"><input id="mk" placeholder="Keywords, comma separated (ai automation, ai course)"><input id="mf" type="file" accept="image/*"><button onclick="addMedia()">Upload</button></div><br><div class="grid">`+
 M.map(m=>`<div class="card"><img src="/dashboard/media/${m.id}"><b>${esc(m.title)}</b><br><small>${esc(m.keywords)}</small><br><button onclick="delMedia(${m.id})" style="background:#b91c1c;margin-top:6px">Delete</button></div>`).join('')+'</div>'}
async function addMedia(){const f=$('#mf').files[0];if(!f)return;await fetch(`/dashboard/api/media?title=${encodeURIComponent($('#mt').value)}&keywords=${encodeURIComponent($('#mk').value)}`,{method:'POST',headers:{'Content-Type':f.type},body:f});drawMedia()}
async function delMedia(id){await fetch('/dashboard/api/media/'+id,{method:'DELETE'});drawMedia()}
setInterval(()=>{if($('#app').style.display==='flex'&&view!=='media')draw()},3000);
fetch('/dashboard/api/chats').then(r=>{if(r.ok){$('#login').style.display='none';$('#app').style.display='flex';init()}});
</script></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"), encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers={"Cache-Control": "no-store"})
    except Exception:
        return HTMLResponse(DASHBOARD_HTML, headers={"Cache-Control": "no-store"})


from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse("/dashboard")


# The Gradio test chat now lives at /test; the main address opens the dashboard.
app = gr.mount_gradio_app(app, demo, path="/test")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
