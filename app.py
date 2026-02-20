"""
app.py — Flask web server for the Vision assistant.
Wraps the core logic from main.py into REST endpoints consumed by the web UI.
"""

import os
import uuid
import datetime
import threading
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ─── Global state (single-user session) ───────────────────────────────────────
_initialized   = False
_init_lock     = threading.Lock()
_user_name     = None
_profile_col   = None
_convo_col     = None
_memory_col    = None
_visual_col    = None
_previous_caption: str | None = None
_last_screenshot_path: str | None = None

# ─── Heavy models — loaded once at startup ────────────────────────────────────
print("Loading HuggingFace login...")
from huggingface_hub import login as hf_login
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("HuggingFace: logged in.")

print("Loading embedding model...")
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

print("Loading BLIP captioning model...")
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
BLIP_MODEL_ID  = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
caption_model  = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
print("BLIP model ready.")

from groq import Groq
bot = Groq(api_key=os.getenv("GROQ_API_KEY"))

import chromadb
shared_client      = chromadb.PersistentClient(path="./chroma_db")
intents_collection = shared_client.get_or_create_collection(name="intents")
profile_client     = chromadb.PersistentClient(path="./chroma_profiles")
convo_client       = chromadb.PersistentClient(path="./chroma_conversations")
visual_client      = chromadb.PersistentClient(path="./chroma_visual")

# ─── Intent templates ──────────────────────────────────────────────────────────
INTENT_TEMPLATES = [
    {"id": "intent_introduce_self",   "text": "let me introduce myself, my name is",        "label": "introduce_self"},
    {"id": "intent_describe_project", "text": "I am working on a project about",             "label": "describe_project"},
    {"id": "intent_ask_explanation",  "text": "can you explain how this works",              "label": "ask_explanation"},
    {"id": "intent_ask_help",         "text": "I need help with",                            "label": "ask_help"},
    {"id": "intent_casual",           "text": "hey how's it going what's up just chatting",  "label": "casual_conversation"},
]

PROFILE_TEMPLATES = {
    "name_statement":      ["my name is John", "you can call me Rahul", "I am called Sam", "people call me Alex", "call me Adi"],
    "project_description": ["I am building an AI assistant", "I am working on a web application project", "my project is about machine learning", "I am developing a mobile app", "I am creating a chatbot"],
    "skill_description":   ["I am good at programming", "my skills include Python and machine learning", "I know how to code in JavaScript", "I am proficient in data science", "I excel at backend development"],
    "weakness_statement":  ["I am weak at mathematics", "I struggle with algorithms", "I find it hard to understand deep learning", "I am not great at design", "I have trouble with calculus"],
    "interest_statement":  ["I am interested in artificial intelligence", "I love reading about technology", "I enjoy playing chess", "my hobby is photography", "I am passionate about robotics"],
}

BASE_SYSTEM_PROMPT = (
    "You are Vision, a personal AI assistant. "
    "Use conversation history to maintain context and help the user "
    "learn, solve problems, and remember details. "
    "If a current visual observation is present, "
    "prioritize it over any past observations or memories. "
    "Do not assume the scene is unchanged. "
    "Only compare current and past visual observations when the user "
    "explicitly asks for comparison."
)

VISION_TRIGGERS   = {"look", "see", "what do you see", "observe"}
_COMPARE_PHRASES  = ["compare", "what changed", "difference", "same as before", "did anything change", "compare with previous"]


# ─── Helper functions (same as main.py) ───────────────────────────────────────

def get_embedding(text: str) -> list:
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))


def detect_intent(embedding: list) -> tuple:
    results = intents_collection.query(query_embeddings=[embedding], n_results=1)
    if results and results["metadatas"] and results["metadatas"][0]:
        label      = results["metadatas"][0][0]["label"]
        distance   = results["distances"][0][0]
        confidence = max(0.0, 1.0 - distance / 2.0)
        return label, confidence
    return "unknown", 0.0


def extract_short_fact(text: str, category: str) -> str:
    prompt = (
        f"Extract the core {category.replace('_', ' ')} information from the following "
        f"sentence in under 6 words. Return ONLY the extracted phrase, nothing else.\n\n"
        f'Sentence: "{text}"'
    )
    try:
        resp = bot.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        fact = resp.choices[0].message.content.strip().strip('"').strip("'")
        return fact if fact else text
    except Exception:
        return text


def store_profile_fact(category, short_fact, profile_col, user_name):
    profile_col.add(
        ids=[str(uuid.uuid4())],
        documents=[short_fact],
        embeddings=[get_embedding(short_fact)],
        metadatas=[{"category": category, "user_name": user_name, "timestamp": datetime.datetime.now().isoformat()}],
    )


def extract_profile_info(text, embedding, profile_col, user_name):
    THRESHOLD = 0.75
    profile_template_embeddings = {
        cat: [get_embedding(p) for p in phrases]
        for cat, phrases in PROFILE_TEMPLATES.items()
    }
    best_category, best_score = None, -1.0
    for category, tmpl_embs in profile_template_embeddings.items():
        max_score = max(cosine_similarity(embedding, te) for te in tmpl_embs)
        if max_score > best_score:
            best_score, best_category = max_score, category
    if best_score < THRESHOLD or best_category is None:
        return
    short_fact = extract_short_fact(text, best_category)
    store_profile_fact(best_category, short_fact, profile_col, user_name)


def get_profile_summary(profile_col) -> str:
    if profile_col.count() == 0:
        return ""
    results = profile_col.get(include=["documents", "metadatas"])
    docs, metas = results.get("documents", []), results.get("metadatas", [])
    if not docs:
        return ""
    lines = [f"- [{m.get('category','fact')}] {d}" for d, m in zip(docs, metas)]
    return "Known user facts:\n" + "\n".join(lines)


def store_conversation_turn(user_text, assistant_text, convo_col, user_name):
    doc = f"User: {user_text}\nAssistant: {assistant_text}"
    convo_col.add(
        ids=[str(uuid.uuid4())],
        documents=[doc],
        embeddings=[get_embedding(user_text)],
        metadatas=[{"user_name": user_name, "timestamp": datetime.datetime.now().isoformat()}],
    )


def get_recent_conversation(convo_col, n=6) -> list:
    total = convo_col.count()
    if total == 0:
        return []
    fetch_n = min(total, max(n, 20))
    results = convo_col.get(limit=fetch_n, include=["documents", "metadatas"])
    docs, metas = results.get("documents", []), results.get("metadatas", [])
    if not docs:
        return []
    paired = sorted(zip(metas, docs), key=lambda x: x[0].get("timestamp", ""))
    recent = paired[-n:]
    messages = []
    for _, doc in recent:
        lines = doc.split("\n", 1)
        if len(lines) == 2:
            user_content = lines[0].removeprefix("User: ")
            asst_content = lines[1].removeprefix("Assistant: ")
        else:
            user_content, asst_content = doc, ""
        messages.append({"role": "user", "content": user_content})
        if asst_content:
            messages.append({"role": "assistant", "content": asst_content})
    return messages


def store_user_message(text, embedding, memory_col, user_name):
    memory_col.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"user_name": user_name, "timestamp": datetime.datetime.now().isoformat()}],
    )


def recall_similar_messages(embedding, memory_col, n=3) -> str:
    total = memory_col.count()
    if total == 0:
        return ""
    results = memory_col.query(query_embeddings=[embedding], n_results=min(n, total))
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""
    return "Relevant past messages from this user:\n" + "\n".join(f"- {d}" for d in docs)


def store_visual_memory(caption, image_path, visual_col, user_name):
    visual_col.add(
        ids=[str(uuid.uuid4())],
        documents=[caption],
        embeddings=[get_embedding(caption)],
        metadatas=[{"user_name": user_name, "image_path": image_path, "timestamp": datetime.datetime.now().isoformat()}],
    )


def recall_visual(query_embedding, visual_col, n=2) -> str:
    total = visual_col.count()
    if total == 0:
        return ""
    results = visual_col.query(query_embeddings=[query_embedding], n_results=min(n, total))
    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""
    lines = "\n".join(f"- {d}" for d in docs)
    return "PAST VISUAL OBSERVATIONS (FOR REFERENCE ONLY):\n" + lines + "\nUse these only if relevant. The current observation is more important."


def capture_frame(save_path="vision_frame.jpg") -> str | None:
    import cv2
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    cv2.imwrite(save_path, frame)
    return save_path


def describe_frame(image_path) -> str:
    raw_image = Image.open(image_path).convert("RGB")
    inputs    = blip_processor(raw_image, return_tensors="pt")
    out       = caption_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(out[0], skip_special_tokens=True)


def is_vision_trigger(text: str) -> bool:
    lowered = text.strip().lower()
    return any(t in lowered for t in VISION_TRIGGERS)


def is_visual_compare_request(text: str) -> bool:
    lowered = text.strip().lower()
    return any(p in lowered for p in _COMPARE_PHRASES)


# ─── Seed intents ──────────────────────────────────────────────────────────────
if intents_collection.count() == 0:
    intents_collection.add(
        ids=[t["id"] for t in INTENT_TEMPLATES],
        documents=[t["text"] for t in INTENT_TEMPLATES],
        embeddings=[get_embedding(t["text"]) for t in INTENT_TEMPLATES],
        metadatas=[{"label": t["label"]} for t in INTENT_TEMPLATES],
    )

# Pre-compute profile template embeddings once at startup
PROFILE_TEMPLATE_EMBEDDINGS = {
    cat: [get_embedding(p) for p in phrases]
    for cat, phrases in PROFILE_TEMPLATES.items()
}


# ─── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/setup", methods=["POST"])
def setup():
    """Initialize per-user ChromaDB collections."""
    global _user_name, _profile_col, _convo_col, _memory_col, _visual_col, _initialized
    data = request.get_json()
    name = data.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "Name is required"}), 400
    with _init_lock:
        _user_name   = name
        _profile_col = profile_client.get_or_create_collection(name=f"profile_{name}")
        _convo_col   = convo_client.get_or_create_collection(name=f"conversation_{name}")
        _memory_col  = convo_client.get_or_create_collection(name=f"memory_{name}")
        _visual_col  = visual_client.get_or_create_collection(name=f"visual_{name}")
        _initialized = True
    return jsonify({"status": "ok", "user": name})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Process one conversation turn. Returns reply + vision info."""
    global _previous_caption, _last_screenshot_path

    if not _initialized:
        return jsonify({"error": "Session not initialized. Please set your name first."}), 400

    data = request.get_json()
    inp  = (data.get("message") or "").strip()
    if not inp:
        return jsonify({"error": "Empty message"}), 400

    vision_used       = False
    screenshot_taken  = False
    caption_out       = None
    live_visual_block = ""
    comparison_block  = ""
    captured_caption  = ""
    captured_path     = ""

    # ── Visual branch ──────────────────────────────────────────────────────────
    if is_vision_trigger(inp):
        frame_path = capture_frame()
        if frame_path:
            caption = describe_frame(frame_path)
            vision_used      = True
            screenshot_taken = True
            caption_out      = caption
            _last_screenshot_path = frame_path

            prev_caption      = _previous_caption
            _previous_caption = caption

            if is_visual_compare_request(inp) and prev_caption is not None:
                comparison_block = (
                    "VISUAL COMPARISON REQUESTED:\n\n"
                    f"Previous observation:\n{prev_caption}\n\n"
                    f"Current observation:\n{caption}\n\n"
                    "Describe what changed or confirm if the scene is the same."
                )

            live_visual_block = (
                "CURRENT VISUAL OBSERVATION (MOST IMPORTANT):\n"
                f"{caption}\n\n"
                "This is the most recent camera frame.\n"
                "It overrides any previous visual observations."
            )
            print(f"  [Vision] live_visual_block built: {caption[:80]}")
            captured_caption = caption
            captured_path    = frame_path
        else:
            live_visual_block = "Visual observation failed: webcam unavailable."
            vision_used = True

    # ── Semantic pipeline ──────────────────────────────────────────────────────
    emb = get_embedding(inp)
    intent, confidence = detect_intent(emb)

    # Profile extraction (reuse pre-computed embeddings)
    THRESHOLD = 0.75
    best_category, best_score = None, -1.0
    for category, tmpl_embs in PROFILE_TEMPLATE_EMBEDDINGS.items():
        max_score = max(cosine_similarity(emb, te) for te in tmpl_embs)
        if max_score > best_score:
            best_score, best_category = max_score, category
    if best_score >= THRESHOLD and best_category:
        short_fact = extract_short_fact(inp, best_category)
        store_profile_fact(best_category, short_fact, _profile_col, _user_name)

    store_user_message(inp, emb, _memory_col, _user_name)

    # ── Build LLM messages ─────────────────────────────────────────────────────
    messages_for_llm = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    profile_summary = get_profile_summary(_profile_col)
    if profile_summary:
        messages_for_llm.append({"role": "system", "content": profile_summary})

    messages_for_llm.extend(get_recent_conversation(_convo_col, n=6))

    memory_block = recall_similar_messages(emb, _memory_col)
    if memory_block:
        messages_for_llm.append({"role": "system", "content": memory_block})

    visual_recall = recall_visual(emb, _visual_col)
    if visual_recall:
        messages_for_llm.append({"role": "system", "content": visual_recall})

    if comparison_block:
        messages_for_llm.append({"role": "system", "content": comparison_block})

    if live_visual_block:
        messages_for_llm.append({"role": "system", "content": live_visual_block})

    messages_for_llm.append({"role": "user", "content": inp})

    # ── Debug: print full prompt so we can verify visual block is present ─────
    print("\n--- PROMPT TO LLM ---")
    for m in messages_for_llm:
        print(m["role"], ":", m["content"][:120])
    print("---------------------\n")

    # ── Groq call ──────────────────────────────────────────────────────────────
    answer = bot.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages_for_llm,
        temperature=0.4,
        max_tokens=300,
    )
    reply = answer.choices[0].message.content

    # ── Persist ────────────────────────────────────────────────────────────────
    store_conversation_turn(inp, reply, _convo_col, _user_name)
    if captured_caption and captured_path:
        store_visual_memory(captured_caption, captured_path, _visual_col, _user_name)

    return jsonify({
        "reply":           reply,
        "intent":          intent,
        "confidence":      round(confidence, 2),
        "vision_used":     vision_used,
        "screenshot_taken": screenshot_taken,
        "caption":         caption_out,
    })


@app.route("/api/screenshot")
def screenshot():
    """Serve the latest captured webcam frame."""
    path = _last_screenshot_path or "vision_frame.jpg"
    if not os.path.exists(path):
        return jsonify({"error": "No screenshot available"}), 404
    # Add no-cache headers so the browser always fetches fresh
    response = send_file(path, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    return response


@app.route("/api/status")
def status():
    return jsonify({
        "initialized": _initialized,
        "user":        _user_name,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
