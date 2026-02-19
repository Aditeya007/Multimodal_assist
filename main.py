import os
import json
import uuid
import datetime
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

# ─── Environment & Clients ────────────────────────────────────────────────────
load_dotenv()
bot = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Embedding Model ──────────────────────────────────────────────────────────
print("Loading embedding model (first run downloads ~80 MB)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")


def get_embedding(text: str) -> list:
    """Return the embedding vector for a piece of text."""
    return embedding_model.encode(text).tolist()


# ─── ChromaDB Setup ───────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
user_memory_collection = chroma_client.get_or_create_collection(name="user_memory")
intents_collection     = chroma_client.get_or_create_collection(name="intents")

# ─── Intent Templates (seed once) ─────────────────────────────────────────────
INTENT_TEMPLATES = [
    {"id": "intent_introduce_self",   "text": "let me introduce myself, my name is",           "label": "introduce_self"},
    {"id": "intent_describe_project", "text": "I am working on a project about",               "label": "describe_project"},
    {"id": "intent_ask_explanation",  "text": "can you explain how this works",                "label": "ask_explanation"},
    {"id": "intent_ask_help",         "text": "I need help with",                              "label": "ask_help"},
    {"id": "intent_casual",           "text": "hey how's it going what's up just chatting",    "label": "casual_conversation"},
]

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
# Each category now has MULTIPLE example phrases for better semantic coverage.
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
# Result: PROFILE_TEMPLATE_EMBEDDINGS[category] = list[embedding_vector]
print("Pre-computing profile template embeddings...")
PROFILE_TEMPLATE_EMBEDDINGS: dict = {
    category: [get_embedding(phrase) for phrase in phrases]
    for category, phrases in PROFILE_TEMPLATES.items()
}
print("Profile template embeddings ready.")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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
    Use the Groq LLM to compress the user's sentence into a short, clean fact
    (under 6 words) that is suitable for storing in the user profile.
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
        return text  # Fallback: keep original text


def extract_profile_info(text: str, embedding: list, user_profile: dict, profile_file: str):
    """
    Compare the user embedding against ALL example embeddings per category.
    Use the MAX similarity score per category, then pick the best category.
    If score > THRESHOLD, compress the fact via LLM and update the profile.
    """
    THRESHOLD = 0.75  # Only store high-confidence matches

    best_category = None
    best_score    = -1.0

    for category, tmpl_embs in PROFILE_TEMPLATE_EMBEDDINGS.items():
        # Score = max similarity across all examples in this category
        max_score = max(cosine_similarity(embedding, tmpl_emb) for tmpl_emb in tmpl_embs)
        if max_score > best_score:
            best_score    = max_score
            best_category = category

    if best_score < THRESHOLD or best_category is None:
        return  # Not confident enough — do not pollute the profile

    # Compress the sentence into a clean short fact via LLM
    short_fact = extract_short_fact(text, best_category)

    # Update the appropriate profile field
    if best_category == "name_statement":
        user_profile["name"] = short_fact
    elif best_category == "project_description":
        if short_fact not in user_profile["projects"]:
            user_profile["projects"].append(short_fact)
    elif best_category == "skill_description":
        user_profile["skill_level"] = short_fact
    elif best_category == "weakness_statement":
        if short_fact not in user_profile["weak_topics"]:
            user_profile["weak_topics"].append(short_fact)
    elif best_category == "interest_statement":
        if short_fact not in user_profile["interests"]:
            user_profile["interests"].append(short_fact)

    # Persist changes
    with open(profile_file, "w") as f:
        json.dump(user_profile, f, indent=2)

    print(f"  [Profile] '{best_category}' → '{short_fact}'  (score={best_score:.2f})")


def store_user_message(text: str, embedding: list, user_name: str):
    """Store a user message in ChromaDB's user_memory collection with metadata."""
    user_memory_collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "user_name": user_name,
            "timestamp": datetime.datetime.now().isoformat(),
        }],
    )


def recall_similar_messages(embedding: list, user_name: str, n: int = 3) -> str:
    """
    Query ChromaDB for the top N most semantically similar past messages
    from this user. Returns a formatted string for use as a system context block.
    Returns an empty string if fewer than 2 messages exist (avoids trivial recalls).
    """
    total = user_memory_collection.count()
    if total < 2:
        return ""

    results = user_memory_collection.query(
        query_embeddings=[embedding],
        n_results=min(n, total),
        where={"user_name": user_name},
    )

    docs = results.get("documents", [[]])[0]
    if not docs:
        return ""

    lines = "\n".join(f"- {doc}" for doc in docs)
    return f"Relevant past messages from this user:\n{lines}"


# ─── User Setup ────────────────────────────────────────────────────────────────
user_name = input("Enter your name: ").strip().lower()

MEMORY_FILE  = f"conversation_{user_name}.json"
PROFILE_FILE = f"user_profile_{user_name}.json"

# Load or create conversation memory
if os.path.exists(MEMORY_FILE):
    try:
        with open(MEMORY_FILE, "r") as f:
            conversation = json.load(f)
    except Exception:
        conversation = [
            {"role": "system", "content": "You are Vision, a personal AI assistant. Use conversation history to maintain context and help the user learn, solve problems, and remember details."}
        ]
else:
    conversation = [
        {"role": "system", "content": "You are Vision, a personal AI assistant. Use conversation history to maintain context and help the user learn, solve problems, and remember details."}
    ]

# Load or create user profile
if os.path.exists(PROFILE_FILE):
    with open(PROFILE_FILE, "r") as f:
        user_profile = json.load(f)
else:
    user_profile = {
        "name":       user_name,
        "interests":  [],
        "projects":   [],
        "skill_level": "",
        "weak_topics": [],
    }
    with open(PROFILE_FILE, "w") as f:
        json.dump(user_profile, f, indent=2)


# ─── Main Conversation Loop ───────────────────────────────────────────────────
print(f"\nVision is ready. Profile: {PROFILE_FILE}")
print("Type 'exit', 'bye', 'sleep', 'shutdown', or 'quit' to end.\n")

while True:
    inp = input("How can I help you today, sir? ").strip()

    if inp.lower() in ["exit", "bye", "sleep", "shutdown", "quit"]:
        print("Pleasure working for you.")
        break

    # ── Semantic Pipeline ──────────────────────────────────────────────────

    # 1. Generate embedding for the current input
    emb = get_embedding(inp)

    # 2. Detect intent via ChromaDB similarity search
    intent, confidence = detect_intent(emb)
    print(f"  [Semantic] intent={intent}  confidence={confidence:.2f}")

    # 3. Extract & update profile info (high-threshold, LLM-compressed)
    extract_profile_info(inp, emb, user_profile, PROFILE_FILE)

    # 4. Store message in ChromaDB before querying (so recall improves over time)
    store_user_message(inp, emb, user_name)

    # 5. Recall top-3 similar past messages and prepend as a system context block
    memory_block = recall_similar_messages(emb, user_name)
    messages_for_llm = conversation.copy()
    if memory_block:
        messages_for_llm.insert(1, {"role": "system", "content": memory_block})

    # ── Groq LLM Call (structure unchanged) ───────────────────────────────
    conversation.append({"role": "user", "content": inp})
    messages_for_llm[-1] = {"role": "user", "content": inp}   # sync last msg

    answer = bot.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages_for_llm,
        temperature=0.4,
        max_tokens=300,
    )
    reply = answer.choices[0].message.content
    print("Vision:", reply)

    conversation.append({"role": "assistant", "content": reply})
    with open(MEMORY_FILE, "w") as f:
        json.dump(conversation, f, indent=2)