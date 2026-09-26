"""
Leaders Academia — RAG Chatbot
Runs as a Hugging Face Space (Gradio SDK).

Pipeline: PDF -> text chunks -> free local embeddings -> FAISS index ->
Gemini (context-grounded answer) -> Gradio chat UI.
"""

import os
import time

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import gradio as gr

# ---------- Configuration ----------
PDF_PATH = "Leaders_Academia_Full_Data.pdf"  # must sit next to app.py in this repo
GEMINI_MODEL_NAME = "gemini-3.8-flash"

HUMAN_HANDOFF_KEYWORDS = [
    "human", "real person", "agent", "representative",
    "insan se baat", "banda se baat", "customer support",
]

# ---------- Load API key from the Space's secret (never hardcode it) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it under this Space's "
        "Settings -> Variables and secrets -> New secret."
    )
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel(GEMINI_MODEL_NAME)


# ---------- Build the knowledge base once, when the Space starts ----------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=800, overlap=150):
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


# ---------- RAG logic ----------
def retrieve_chunks(query, top_k=4):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]


def wants_human(text):
    text = text.lower()
    return any(keyword in text for keyword in HUMAN_HANDOFF_KEYWORDS)


def rag_answer(user_question, max_retries=3):
    # Direct handoff — skip the model entirely for this case
    if wants_human(user_question):
        return (
            "Zaroor! Aap hamari team se seedha rabta kar sakte hain:\n"
            "Phone: (051) 9876543\n"
            "WhatsApp Support: +92 311 1534344"
        )

    context_chunks = retrieve_chunks(user_question, top_k=4)
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""You are the official AI assistant for Leaders Academia.

LANGUAGE RULE: Always reply in the SAME language and script the user used in
their message (English, Urdu script, or Roman Urdu). Never force one language
if the user wrote in a different one.

CONVERSATION RULE: For greetings and small talk (e.g. "how are you", "hi",
"thanks") reply naturally and warmly like a human — never say information is
unavailable for these.

FACTUAL RULE: For questions about courses, instructors, pricing, schedules or
the platform, answer ONLY using the CONTEXT below. If the answer is not in the
CONTEXT, say clearly that this information is not available — never guess.

CONTEXT:
{context}

USER MESSAGE: {user_question}

Reply clearly and concisely."""

    for attempt in range(max_retries):
        try:
            response = gen_model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_text = str(e)
            if "429" in error_text or "quota" in error_text.lower():
                if attempt < max_retries - 1:
                    time.sleep(15)  # back off and retry once on rate limit
                    continue
                return (
                    "Maaf kijiye, is waqt AI system busy hai (free usage "
                    "limit lag gayi hai). Thori dair mein dobara try karein, "
                    "ya seedha rabta karein: +92 311 1534344 (WhatsApp)."
                )
            return "Kuch masla aa gaya jawab generate karte waqt, dobara koshish karein."


# ---------- Gradio frontend ----------
def chat_fn(message, history):
    return rag_answer(message)


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Leaders Academia — AI Assistant",
    description="Courses, instructors aur platform ke baare mein kuch bhi puchein.",
)

if __name__ == "__main__":
    # Render (and most free hosts) assign a PORT via env var and expect the
    # app to bind to 0.0.0.0 so it's reachable from outside the container.
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
