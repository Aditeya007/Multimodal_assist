import os
import uuid
import datetime
import numpy as np
from groq import Groq
from dotenv import load_dotenv

# ─── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

# ─── HuggingFace Login ────────────────────────────────────────────────────────
# Authenticates with HF Hub using the token from .env / environment.
# This enables authenticated model downloads and faster caching.
from huggingface_hub import login as hf_login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("HuggingFace: logged in successfully.")
else:
    print("HuggingFace: no HF_TOKEN found — proceeding unauthenticated.")

# ─── Embedding Model (sentence-transformers) ──────────────────────────────────
from sentence_transformers import SentenceTransformer

print("Loading embedding model (all-MiniLM-L6-v2)...")
# normalize_embeddings=True ensures unit-length vectors so cosine similarity
# is computed correctly without extra normalisation steps.
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

# ─── BLIP Vision Model ────────────────────────────────────────────────────────
# Loaded once at startup. Used to generate captions from webcam frames.
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("Loading BLIP image captioning model (first run downloads ~1 GB)...")
BLIP_MODEL_ID  = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
caption_model  = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
print("BLIP model ready.")

# ─── Groq Client ──────────────────────────────────────────────────────────────
bot = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── ChromaDB — Three Folder-Based Persistent Clients ────────────────────────
# Each domain gets its own storage folder for clean separation.
import chromadb

# Shared client for static intent templates (unchanged location)
shared_client = chromadb.PersistentClient(path="./chroma_db")
intents_collection = shared_client.get_or_create_collection(name="intents")

# Per-domain clients
profile_client = chromadb.PersistentClient(path="./chroma_profiles")
convo_client   = chromadb.PersistentClient(path="./chroma_conversations")
visual_client  = chromadb.PersistentClient(path="./chroma_visual")

# ─── Intent Templates (seed once into shared client) ─────────────────────────
INTENT_TEMPLATES = [
    {"id": "intent_introduce_self",   "text": "let me introduce myself, my name is",           "label": "introduce_self"},
    {"id": "intent_describe_project", "text": "I am working on a project about",               "label": "describe_project"},
    {"id": "intent_ask_explanation",  "text": "can you explain how this works",                "label": "ask_explanation"},
    {"id": "intent_ask_help",         "text": "I need help with",                              "label": "ask_help"},
    {"id": "intent_casual",           "text": "hey how's it going what's up just chatting",    "label": "casual_conversation"},
]


def get_embedding(text: str) -> list:
    """Return the normalized embedding vector for a piece of text."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


if intents_collection.count() == 0:
    print("Seeding intent templates into ChromaDB...")
    intents_collection.add(
        ids=[t["id"] for t in INTENT_TEMPLATES],
        documents=[t["text"] for t in INTENT_TEMPLATES],
        embeddings=[get_embedding(t["text"]) for t in INTENT_TEMPLATES],
        metadatas=[{"label": t["label"]} for t in INTENT_TEMPLATES],
    )
    print(f"  Seeded {len(INTENT_TEMPLATES)} intent templates.")

# ─── Profile Category Templates (multi-example per category) ─────────────────
PROFILE_TEMPLATES: dict = {
    "name_statement": [
        "my name is John",
        "you can call me Rahul",
        "I am called Sam",
        "people call me Alex",
        "call me Adi",
    ],
    "project_description": [
        "I am building an AI assistant",
        "I am working on a web application project",
        "my project is about machine learning",
        "I am developing a mobile app",
        "I am creating a chatbot",
    ],
    "skill_description": [
        "I am good at programming",
        "my skills include Python and machine learning",
        "I know how to code in JavaScript",
        "I am proficient in data science",
        "I excel at backend development",
    ],
    "weakness_statement": [
        "I am weak at mathematics",
        "I struggle with algorithms",
        "I find it hard to understand deep learning",
        "I am not great at design",
        "I have trouble with calculus",
    ],
    "interest_statement": [
        "I am interested in artificial intelligence",
        "I love reading about technology",
        "I enjoy playing chess",
        "my hobby is photography",
        "I am passionate about robotics",
    ],
}

# Pre-compute embeddings for EACH example phrase per category.
print("Pre-computing profile template embeddings...")
PROFILE_TEMPLATE_EMBEDDINGS: dict = {
    category: [get_embedding(phrase) for phrase in phrases]
    for category, phrases in PROFILE_TEMPLATES.items()
}
print("Profile template embeddings ready.")

# Keywords that trigger the visual perception branch
VISION_TRIGGERS = {"look", "see", "what do you see", "observe"}

# Base system prompt — extended with two visual-priority safety rules:
#  (a) always prefer the live observation over stored memories.
#  (b) only compare past vs current when the user explicitly asks (Part 5).
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors.
    The 1e-10 epsilon prevents divide-by-zero for zero-norm vectors.
    """
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def detect_intent(embedding: list) -> tuple:
    """Query the intents collection and return (label, confidence 0-1)."""
    results = intents_collection.query(query_embeddings=[embedding], n_results=1)
    if results and results["metadatas"] and results["metadatas"][0]:
        label      = results["metadatas"][0][0]["label"]
        distance   = results["distances"][0][0]
        confidence = max(0.0, 1.0 - distance / 2.0)
        return label, confidence
    return "unknown", 0.0


def extract_short_fact(text: str, category: str) -> str:
    """
    Compress the user's sentence into a short clean fact (≤6 words) using Groq.
    Falls back to the original text if the API call fails.
    """
    prompt = (
        f"Extract the core {category.replace('_', ' ')} information from the following "
        f"sentence in under 6 words. Return ONLY the extracted phrase, nothing else.\n\n"
        f'Sentence: "{text}"'
    )
    try:
        response = bot.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        fact = response.choices[0].message.content.strip().strip('"').strip("'")
        return fact if fact else text
    except Exception:
        return text


# ─────────────────────────────────────────────────────────────────────────────
# PART 6 — PROFILE FACTS STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def store_profile_fact(category: str, short_fact: str, profile_col, user_name: str):
    """
    Store a single extracted profile fact into the profile collection.
    Each fact is one document; duplicates are allowed (latest wins in summaries).
    """
    profile_col.add(
        ids=[str(uuid.uuid4())],
        documents=[short_fact],
        embeddings=[get_embedding(short_fact)],
        metadatas=[{
            "category":  category,
            "user_name": user_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }],
    )
    print(f"  [Profile] '{category}' → '{short_fact}'")


def extract_profile_info(text: str, embedding: list, profile_col, user_name: str):
    """
    Compare the user embedding against ALL example embeddings per category.
    Use the MAX similarity score per category. Pick the best category.
    If score > 0.75 threshold, compress via LLM and store in ChromaDB.
    """
    THRESHOLD = 0.75

    best_category = None
    best_score    = -1.0

    for category, tmpl_embs in PROFILE_TEMPLATE_EMBEDDINGS.items():
        max_score = max(cosine_similarity(embedding, tmpl_emb) for tmpl_emb in tmpl_embs)
        if max_score > best_score:
            best_score    = max_score
            best_category = category

    if best_score < THRESHOLD or best_category is None:
        return

    short_fact = extract_short_fact(text, best_category)
    store_profile_fact(best_category, short_fact, profile_col, user_name)


# ─────────────────────────────────────────────────────────────────────────────
# PART 7 — PROFILE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def get_profile_summary(profile_col) -> str:
    """
    Retrieve all stored profile facts for this user from ChromaDB and
    format them into a system-context block.
    Returns an empty string if no facts have been stored yet.
    """
    total = profile_col.count()
    if total == 0:
        return ""

    results = profile_col.get(include=["documents", "metadatas"])
    docs      = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not docs:
        return ""

    lines = []
    for doc, meta in zip(docs, metadatas):
        category = meta.get("category", "fact")
        lines.append(f"- [{category}] {doc}")

    return "Known user facts:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — CONVERSATION STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def store_conversation_turn(user_text: str, assistant_text: str, convo_col, user_name: str):
    """
    Store one full conversation turn (user + assistant) as a single document.
    The embedding is generated from the user's text for semantic retrieval later.
    """
    document = f"User: {user_text}\nAssistant: {assistant_text}"
    convo_col.add(
        ids=[str(uuid.uuid4())],
        documents=[document],
        embeddings=[get_embedding(user_text)],
        metadatas=[{
            "user_name": user_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }],
    )


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — CONVERSATION RECALL
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_conversation(convo_col, n: int = 6) -> list:
    """
    Retrieve the most recent N conversation turns from ChromaDB, sorted by
    timestamp ascending (oldest first), and expand each turn into two message
    dicts suitable for the Groq API.

    Returns a list of {"role": ..., "content": ...} dicts.
    """
    total = convo_col.count()
    if total == 0:
        return []

    # Fetch all stored turns (with metadata for sorting)
    fetch_n = min(total, max(n, 20))  # fetch extra so we can sort then trim
    results = convo_col.get(
        limit=fetch_n,
        include=["documents", "metadatas"],
    )

    docs      = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not docs:
        return []

    # Sort by timestamp ascending (oldest first)
    paired = sorted(zip(metadatas, docs), key=lambda x: x[0].get("timestamp", ""))

    # Take the last N turns
    recent = paired[-n:]

    messages = []
    for _, doc in recent:
        # Each doc is "User: ...\nAssistant: ..."
        lines = doc.split("\n", 1)
        if len(lines) == 2:
            user_line, asst_line = lines
            user_content = user_line.removeprefix("User: ")
            asst_content = asst_line.removeprefix("Assistant: ")
        else:
            user_content = doc
            asst_content = ""

        messages.append({"role": "user",      "content": user_content})
        if asst_content:
            messages.append({"role": "assistant", "content": asst_content})

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# PART — SEMANTIC MEMORY (user messages only, for episodic recall)
# ─────────────────────────────────────────────────────────────────────────────

def store_user_message(text: str, embedding: list, memory_col, user_name: str):
    """Store a single user message in the semantic memory collection."""
    memory_col.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "user_name": user_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }],
    )


def recall_similar_messages(embedding: list, memory_col, n: int = 3) -> str:
    """
    Query the semantic memory collection for top-N similar past messages.
    Returns a formatted string for injection as a system message, or "" if empty.
    """
    total = memory_col.count()
    if total == 0:
        return ""

    results = memory_col.query(
        query_embeddings=[embedding],
        n_results=min(n, total),
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    lines = "\n".join(f"- {doc}" for doc in docs)
    return f"Relevant past messages from this user:\n{lines}"


# ─────────────────────────────────────────────────────────────────────────────
# PART 8 — VISUAL MEMORY STORAGE
# ─────────────────────────────────────────────────────────────────────────────

def store_visual_memory(caption: str, image_path: str, visual_col, user_name: str):
    """
    Store a BLIP-generated caption into the visual memory collection.
    The embedding is computed from the caption text for semantic recall.
    """
    visual_col.add(
        ids=[str(uuid.uuid4())],
        documents=[caption],
        embeddings=[get_embedding(caption)],
        metadatas=[{
            "user_name":  user_name,
            "image_path": image_path,
            "timestamp":  datetime.datetime.now().isoformat(),
        }],
    )
    print(f"  [Visual Memory] Stored caption for {image_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 9 — VISUAL RECALL
# ─────────────────────────────────────────────────────────────────────────────

def recall_visual(query_embedding: list, visual_col, n: int = 2) -> str:
    """
    Query the visual memory collection for semantically similar past observations.

    Part 2 — Returns a clearly labelled PAST VISUAL OBSERVATIONS block so the
    LLM never confuses stored memories with the live camera feed.  The live
    caption is never included here; it is injected separately as live_visual_block.

    Returns a formatted string for injection as a system message, or "" if empty.
    """
    total = visual_col.count()
    if total == 0:
        return ""

    results = visual_col.query(
        query_embeddings=[query_embedding],
        n_results=min(n, total),
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    lines = "\n".join(f"- {doc}" for doc in docs)
    # Explicitly label these as PAST observations so the LLM treats them as
    # background reference, not as the current state of the world.
    return (
        "PAST VISUAL OBSERVATIONS (FOR REFERENCE ONLY):\n"
        + lines
        + "\nUse these only if relevant. The current observation is more important."
    )


# ─────────────────────────────────────────────────────────────────────────────
# VISION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame(save_path: str = "vision_frame.jpg") -> str | None:
    """
    Open the default webcam, capture one frame, save it to disk, and
    release the camera immediately. Returns the file path on success, None on failure.
    """
    import cv2
    # cv2.CAP_DSHOW (DirectShow) improves startup speed and reliability on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set a standard resolution to avoid per-device default quirks.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("  [Vision] Could not open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("  [Vision] Failed to capture frame.")
        return None

    cv2.imwrite(save_path, frame)
    print(f"  [Vision] Frame saved → {save_path}")
    return save_path


def describe_frame(image_path: str) -> str:
    """
    Load an image from disk, run it through the BLIP captioning model,
    and return the generated caption string.
    """
    raw_image = Image.open(image_path).convert("RGB")
    inputs    = blip_processor(raw_image, return_tensors="pt")
    out       = caption_model.generate(**inputs, max_new_tokens=50)
    caption   = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def is_vision_trigger(text: str) -> bool:
    """Return True if the user's input matches any visual perception trigger."""
    lowered = text.strip().lower()
    return any(trigger in lowered for trigger in VISION_TRIGGERS)


# Part 2 — Phrases that signal the user wants a before/after comparison.
_COMPARE_PHRASES = [
    "compare",
    "what changed",
    "difference",
    "same as before",
    "did anything change",
    "compare with previous",
]


def is_visual_compare_request(text: str) -> bool:
    """Return True if the user explicitly asks to compare the current frame
    with the previous one.  Comparison blocks are ONLY injected when this
    returns True — keeping normal vision queries clean.
    """
    lowered = text.strip().lower()
    return any(phrase in lowered for phrase in _COMPARE_PHRASES)


# ─────────────────────────────────────────────────────────────────────────────
# USER SETUP — Per-user collections in the correct clients
# ─────────────────────────────────────────────────────────────────────────────
user_name = input("Enter your name: ").strip().lower()

# Create (or open) all four per-user collections
profile_col = profile_client.get_or_create_collection(name=f"profile_{user_name}")
convo_col   = convo_client.get_or_create_collection(name=f"conversation_{user_name}")
memory_col  = convo_client.get_or_create_collection(name=f"memory_{user_name}")
visual_col  = visual_client.get_or_create_collection(name=f"visual_{user_name}")

print(f"\nVision is ready. User: '{user_name}'")
print("Collections → profile_col, convo_col, memory_col, visual_col")
print("Camera triggers: 'look', 'see', 'what do you see', 'observe'")
print("Type 'exit', 'bye', 'sleep', 'shutdown', or 'quit' to end.\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONVERSATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

# `_previous_caption` stores the last successfully captured visual observation
# so we can compare frames. None means no frame has been captured yet this session.
# Using None (not '') lets us distinguish 'never seen a frame' from 'saw a blank'.
_previous_caption: str | None = None

while True:
    inp = input("How can I help you today, sir? ").strip()

    if inp.lower() in ["exit", "bye", "sleep", "shutdown", "quit"]:
        print("Pleasure working for you.")
        break

    # ── Visual Perception Branch ──────────────────────────────────────────
    # live_visual_block  — labelled CURRENT VISUAL OBSERVATION; always injected
    #                      after memory blocks but before the user message.
    # comparison_block   — only built when the user explicitly asks to compare
    #                      (Part 3); empty string otherwise.
    # captured_caption   — held so we can persist to visual memory after the
    #                      LLM call; never mixed into comparison/live blocks.
    live_visual_block = ""   # Empty when camera not triggered this turn
    comparison_block  = ""   # Part 3: empty unless compare intent detected
    captured_caption  = ""   # Persisted to ChromaDB after the LLM call
    captured_path     = ""

    if is_vision_trigger(inp):
        print("  [Vision] Activating webcam...")
        frame_path = capture_frame()
        if frame_path:
            caption = describe_frame(frame_path)
            print(f"  [Vision] Caption: {caption}")

            # Snapshot the previous caption BEFORE updating the tracker
            # so we can use it in the comparison block for this turn.
            # _previous_caption lives at module scope — no 'global' needed here.
            if caption is not None:
                prev_caption      = _previous_caption   # may be None on first run
                _previous_caption = caption             # update for next turn

            # Part 3 — Build comparison_block ONLY when the user explicitly asks.
            # This keeps normal vision queries clean and uncluttered.
            if is_visual_compare_request(inp) and prev_caption is not None:
                comparison_block = (
                    "VISUAL COMPARISON REQUESTED:\n\n"
                    f"Previous observation:\n{prev_caption}\n\n"
                    f"Current observation:\n{caption}\n\n"
                    "Describe what changed or confirm if the scene is the same."
                )
                print("  [Vision] Comparison block built (user requested comparison)")

            # Build the live observation block regardless of comparison intent.
            # Scene-change notice prepended only when captions differ.
            scene_change_notice = ""
            if prev_caption is not None and caption.strip() != prev_caption.strip():
                scene_change_notice = "The scene has changed since the last observation.\n"

            live_visual_block = (
                f"{scene_change_notice}"
                f"CURRENT VISUAL OBSERVATION (MOST IMPORTANT):\n"
                f"{caption}\n\n"
                f"This is the most recent camera frame.\n"
                f"It overrides any previous visual observations."
            )

            captured_caption = caption
            captured_path    = frame_path
        else:
            live_visual_block = "Visual observation failed: webcam unavailable."

    # ── Semantic Pipeline ────────────────────────────────────────────────
    # 1. Generate embedding for the current input
    emb = get_embedding(inp)

    # 2. Detect intent via ChromaDB similarity search
    intent, confidence = detect_intent(emb)
    print(f"  [Semantic] intent={intent}  confidence={confidence:.2f}")

    # 3. Extract & store profile facts (if input matches a category)
    extract_profile_info(inp, emb, profile_col, user_name)

    # 4. Store user message in semantic memory (for future recall)
    store_user_message(inp, emb, memory_col, user_name)

    # ── Build LLM Context — enforced ordering (Parts 1–5) ────────────────
    #
    # Order rationale (ensures LLM sees the freshest information last):
    #   1. base system prompt   — identity + visual-priority + comparison rules
    #   2. profile summary      — known facts about the user
    #   3. recent conversation  — last N turns for local context
    #   4. semantic recall      — episodic memory (similar past messages)
    #   5. visual memory block  — PAST visual observations (reference only)
    #   6. comparison_block     — side-by-side prev vs current (ONLY on request)
    #   7. live_visual_block    — CURRENT camera observation (highest priority)
    #   8. user message         — what the user said this turn
    #
    # comparison_block (step 6) is "" unless is_visual_compare_request() fired,
    # so normal vision turns are completely unaffected.

    # 1. Base system prompt (visual-priority + comparison-only rules from Part 5)
    messages_for_llm = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    # 2. Profile summary (all known facts about the user)
    profile_summary = get_profile_summary(profile_col)
    if profile_summary:
        messages_for_llm.append({"role": "system", "content": profile_summary})

    # 3. Recent conversation turns from ChromaDB
    recent_turns = get_recent_conversation(convo_col, n=6)
    messages_for_llm.extend(recent_turns)

    # 4. Semantic recall block — similar past user messages (episodic memory)
    memory_block = recall_similar_messages(emb, memory_col)
    if memory_block:
        messages_for_llm.append({"role": "system", "content": memory_block})

    # 5. Visual memory block — PAST visual observations, labelled reference-only;
    #    the live caption is never mixed in here.
    visual_recall = recall_visual(emb, visual_col)
    if visual_recall:
        messages_for_llm.append({"role": "system", "content": visual_recall})

    # 6. Comparison block (Part 3) — injected ONLY when the user explicitly asks
    #    to compare (e.g. "compare", "what changed", "did anything change").
    #    Empty string on all other turns so normal vision is unaffected.
    if comparison_block:
        messages_for_llm.append({"role": "system", "content": comparison_block})

    # 7. Live visual block — CURRENT camera observation.
    #    Placed after the comparison block so the LLM's freshest context is
    #    always the live feed, not the side-by-side summary.
    if live_visual_block:
        messages_for_llm.append({"role": "system", "content": live_visual_block})

    # 8. Current user message — always last
    messages_for_llm.append({"role": "user", "content": inp})

    # ── Groq LLM Call ─────────────────────────────────────────────────────
    answer = bot.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages_for_llm,
        temperature=0.4,
        max_tokens=300,
    )
    reply = answer.choices[0].message.content
    print("Vision:", reply)

    # ── Persist Turn to ChromaDB (no JSON) ───────────────────────────────
    # Store the full conversation turn
    store_conversation_turn(inp, reply, convo_col, user_name)

    # Store visual memory if camera was used this turn
    if captured_caption and captured_path:
        store_visual_memory(captured_caption, captured_path, visual_col, user_name)