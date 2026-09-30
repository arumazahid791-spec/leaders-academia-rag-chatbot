"""
Leaders Academia — RAG Chatbot
Runs on Railway (or any host with a Procfile-style start command).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> Gradio chat UI.
"""

import asyncio
import os
import time

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types
from groq import Groq
import gradio as gr
import requests
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

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
    "E-commerce",
    "Digital Marketing",
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


def generate_text(prompt):
    """Try Gemini first; if it's busy, out of quota or slow, fall back to Groq
    right away — same prompt goes to both, so the tone and style of the reply
    stays the same no matter which one actually answers."""
    if gemini_is_up():
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME, contents=prompt
            )
            if response.text:
                return response.text
            print("Gemini returned an empty reply, falling back to Groq.")
        except Exception as e:
            print(f"Gemini failed, falling back to Groq: {e}")
            if is_transient_error(e):
                mark_gemini_down()

    if groq_client:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Groq also failed: {e}")
    else:
        print("GROQ_API_KEY is not set — no backup available, only Gemini was tried.")

    return (
        "Is waqt thora zyada rush hai is liye jawab dene mein masla ho raha "
        f"hai. Chand minute mein dobara message kar dein, ya seedha hamari "
        f"team se baat kar lein: {TEAM_HEAD_NUMBER}"
    )


def rag_answer(user_question, voice=False):
    # Direct handoff — skip the model entirely for this case
    if wants_human(user_question):
        return (
            f"Zaroor! Aap hamari team head se seedha rabta kar sakte hain: "
            f"{TEAM_HEAD_NUMBER}"
        )

    context_chunks = retrieve_chunks(user_question, top_k=6)
    context = "\n\n---\n\n".join(context_chunks)

    voice_rule = ""
    if voice:
        voice_rule = """
VOICE RULE: This reply will be spoken aloud as a voice note. Keep it short
(2-3 sentences), warm and conversational. No lists, no bullet points, no
emojis, no symbols. If the user spoke Urdu or Hindi, reply in Urdu script
(اردو). If the user spoke English, reply in English.
"""

    prompt = f"""You are the official AI assistant for Leaders Academia, chatting with
someone on WhatsApp. Your tone is professional yet conversational — clear,
warm, and easy to talk to, like a knowledgeable team member, not a stiff
formal bot and not overly casual either. Keep this tone consistent in every
reply.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

SCOPE RULE: You only help with Leaders Academia — its courses, instructors,
fees, schedules and platform. Friendly small talk is welcome (greetings,
"how are you", thanks, goodbye) — answer those warmly and briefly, then
gently steer back to how you can help with Leaders Academia. If someone asks
for something unrelated to Leaders Academia (recipes, coding help, general
knowledge, news, homework, etc.), politely say you can only help with
Leaders Academia and invite them to ask about the courses or services. Do NOT
provide the unrelated information, not even briefly.

LEADERS ACADEMIA RULE: For questions about courses, instructors, pricing,
schedules or the platform, answer using the CONTEXT below as if it's simply
what you know — speak naturally and confidently, the way a helpful human
staff member would. Never mention "context", "the information provided",
"based on the available data" or anything that reveals you're reading from a
document.

WHEN YOU DON'T KNOW: If a Leaders Academia question isn't answered by the
CONTEXT, don't guess or invent details. Instead say naturally, in your own
words, that you don't have that detail on hand and give them this number to
reach the team head directly: {TEAM_HEAD_NUMBER}

COMPLETE COURSE LIST (authoritative — this is every course we offer, already
in the right order to mention them in): every time the user asks what
courses are offered, or wants a list of all courses, use exactly this list,
in exactly this order. Don't skip any, don't invent extra ones, don't
reorder them yourself:
{COURSE_LIST_TEXT}
{voice_rule}
CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply clearly, briefly, and naturally — like a real person texting back, not
a formal report."""

    return generate_text(prompt)


# ---------- Gradio frontend ----------
def chat_fn(message, history):
    return rag_answer(message)


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
            if is_overloaded_error(e) and attempt == 0:
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


SINGLE_VOICE = "ur-PK-UzmaNeural"  # one consistent voice, used regardless of language


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
GEMINI_TTS_VOICE = os.environ.get("GEMINI_TTS_VOICE", "Sulafat")
GEMINI_TTS_STYLE = (
    "professional yet conversational — calm, clear, and warm, like an "
    "experienced team member speaking on a call. Not overly casual, not stiff."
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


@app.get("/webhook")
def verify_webhook(request: Request):
    """Meta calls this once, when you save the webhook URL, to confirm you
    own this server. It must echo back the 'hub.challenge' value."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
        return PlainTextResponse(challenge)
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_whatsapp_message(request: Request):
    """Meta calls this every time a user sends a WhatsApp message."""
    data = await request.json()
    try:
        entry = data["entry"][0]
        change = entry["changes"][0]["value"]
        messages = change.get("messages")

        if messages:
            message = messages[0]
            message_id = message.get("id")
            from_number = message["from"]  # sender's WhatsApp number
            msg_type = message.get("type")
            print(f"Incoming WhatsApp {msg_type} message from: {from_number}")

            if message_id in processed_message_ids:
                print(f"Duplicate delivery of message {message_id}, skipping.")
                return {"status": "duplicate, skipped"}
            processed_message_ids.add(message_id)

            user_text = None
            is_voice_message = False

            if msg_type == "text":
                user_text = message.get("text", {}).get("body", "")
            elif msg_type == "audio":
                is_voice_message = True
                media_id = message["audio"]["id"]
                try:
                    audio_bytes, mime_type = download_whatsapp_media(media_id)
                    user_text = transcribe_audio(audio_bytes, mime_type)
                    print(f"Transcribed voice message: {user_text}")
                except Exception as e:
                    print(f"Voice transcription failed after retries: {e}")
                    send_whatsapp_message(
                        from_number,
                        "Yeh voice note samajhne mein masla ho raha hai. "
                        "Dobara bhej kar dekh lein, ya text mein likh dein, "
                        f"ya seedha rabta karein: {TEAM_HEAD_NUMBER}",
                    )
                    user_text = None

            if user_text:
                reply_text = rag_answer(user_text, voice=is_voice_message)

                if is_voice_message:
                    # Voice in, voice out — feels like a real conversation
                    voice_sent = False
                    try:
                        audio, mime_type, filename = await synthesize_reply_audio(reply_text)
                        voice_sent = send_whatsapp_voice(from_number, audio, mime_type, filename)
                    except Exception as e:
                        print(f"Voice reply failed: {e}")
                    if not voice_sent:
                        # Never leave the user without an answer
                        send_whatsapp_message(from_number, reply_text)
                else:
                    send_whatsapp_message(from_number, reply_text)
    except Exception as e:
        print("Error processing incoming WhatsApp message:", e)

    # Always return 200 quickly so Meta doesn't retry/resend the same message
    return {"status": "received"}


# Mount the Gradio chat UI at "/" for browser-based testing, alongside the
# WhatsApp webhook routes above.
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
