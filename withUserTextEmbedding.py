import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pymongo import MongoClient
import uuid
from datetime import datetime, timedelta
import numpy as np
import openai
import logging
import spacy
from nltk.corpus import wordnet
import nltk
from cachetools import TTLCache
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from newEntry import clear_database, verify_data ,populate_users, populate_conversations, populate_journals  # Import from data entry script
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in .env file")
    raise ValueError("OPENAI_API_KEY not found")
if not MONGODB_URI:
    logger.error("MONGODB_URI not found in .env file")
    raise ValueError("MONGODB_URI not found")
logger.info(f"Loaded OPENAI_API_KEY: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")
logger.info(f"Loaded MONGODB_URI: {MONGODB_URI[:30]}...")

# MongoDB Atlas setup
logger.info("Connecting to MongoDB Atlas")
client = MongoClient(
    MONGODB_URI,
    tls=True,
    tlsAllowInvalidCertificates=True
)
db = client["LF"]
users_collection = db["users"]
conversations = db["conversations"]
journal_collection = db["journal_entries"]
embeddings_collection = db["embeddings"]
personalities_collection = db["personalities"]
errors_collection = db["errors"]
saved_greetings = db["saved_greetings"]

# Ensure indexes
logger.info("Creating indexes for MongoDB collections")
conversations.create_index([("user_id", 1), ("timestamp", -1)])
conversations.create_index([("speaker_id", 1), ("target_id", 1), ("timestamp", -1)])
conversations.create_index([("content", "text")])
journal_collection.create_index([("user_id", 1), ("timestamp", -1)])
journal_collection.create_index([("content", "text")])
embeddings_collection.create_index([("item_id", 1), ("item_type", 1)])
personalities_collection.create_index([("user_id", 1)])
errors_collection.create_index([("timestamp", -1)])
saved_greetings.create_index([("target_id", 1), ("bot_role", 1), ("timestamp", -1)])

# LangChain and FAISS setup
logger.info("Initializing LangChain OpenAI embeddings")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Initialize FAISS vector store (in-memory for this example; persist to disk in production)
faiss_store = None

def initialize_faiss_store():
    global faiss_store
    logger.info("Initializing FAISS vector store")
    embeddings_data = list(embeddings_collection.find())
    documents = []
    for emb in embeddings_data:
        try:
            item_id = emb.get("item_id")
            item_type = emb.get("item_type")
            if not item_id or not item_type:
                logger.warning(f"Deleting invalid embedding: {emb}")
                embeddings_collection.delete_one({"_id": emb["_id"]})
                continue

            collection = conversations if item_type == "conversation" else journal_collection
            id_field = "conversation_id" if item_type == "conversation" else "entry_id"
            doc = collection.find_one({id_field: item_id})
            if not doc:
                logger.warning(f"Deleting orphaned embedding: item_id={item_id}, item_type={item_type}")
                embeddings_collection.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            content = emb.get("content", doc.get("content", ""))
            if not content:
                logger.warning(f"No content for item_id: {item_id}, item_type={item_type}")
                embeddings_collection.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            user_id = emb.get("user_id", [])
            if not user_id:
                logger.warning(f"No user_id for item_id: {item_id}")
                embeddings_collection.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            speaker_name = emb.get("speaker_name")
            if not speaker_name:
                speaker_name = users_collection.find_one({"user_id": user_id[0]})["name"] if user_id else "Unknown"

            target_name = emb.get("target_name")
            if not target_name and len(user_id) > 1:
                target_name = users_collection.find_one({"user_id": user_id[1]})["name"] if len(user_id) > 1 else None

            metadata = {
                "item_id": item_id,
                "item_type": item_type,
                "user_id": user_id,
                "speaker_id": emb.get("speaker_id", user_id[0] if item_type == "journal" else None),
                "target_id": emb.get("target_id", None),
                "speaker_name": speaker_name,
                "target_name": target_name,
                "timestamp": emb["timestamp"]
            }
            documents.append(Document(page_content=content, metadata=metadata))
            logger.debug(f"Added to FAISS: item_id={item_id}, item_type={item_type}")
        except Exception as e:
            logger.warning(f"Error processing embedding: {str(e)}")
            embeddings_collection.delete_one({"_id": emb["_id"]})
            continue

    faiss_store = FAISS.from_documents(documents, embeddings) if documents else FAISS.from_texts(["empty"], embeddings)
    logger.info(f"Populated FAISS with {len(documents)} documents")

initialize_faiss_store()
# OpenAI setup
logger.info("OpenAI API key loaded")
client2 = openai.OpenAI(api_key=OPENAI_API_KEY)

# FastAPI app
app = FastAPI(title="Chatbot AI Twin API")

# API key for basic authentication
API_KEY = "your-secure-api-key"

# Cache for embeddings
embedding_cache = TTLCache(maxsize=1000, ttl=3600)

# Pydantic models for request/response
class MessageRequest(BaseModel):
    speaker_id: str
    target_id: str
    bot_role: str
    user_input: str

class MessageResponse(BaseModel):
    response: str
    error: Optional[str] = None

def preprocess_input(user_input: str) -> str:
    logger.debug(f"Preprocessing input: {user_input[:50]}...")
    try:
        doc = nlp(user_input)
        key_terms = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "VERB"] and not token.is_stop]
        extra_terms = []
        for term in key_terms:
            synsets = wordnet.synsets(term)
            synonyms = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != term and len(synonym.split()) <= 2:
                        synonyms.add(synonym)
                extra_terms.extend(list(synonyms)[:3])
        if extra_terms:
            user_input += " " + " ".join(set(extra_terms[:10]))
        return user_input
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        return user_input

def get_recent_conversation_history(speaker_id: str, target_id: str, limit: int = 6) -> List[dict]:
    logger.info(f"Retrieving recent conversation history for speaker={speaker_id}, target={target_id}")
    history = []
    pipeline = [
        {
            "$match": {
                "user_id": {"$all": [speaker_id, target_id], "$size": 2},
                "$or": [
                    {"speaker_id": speaker_id, "target_id": target_id},
                    {"speaker_id": target_id, "target_id": speaker_id}
                ]
            }
        },
        {"$sort": {"timestamp": -1}},
        {"$limit": limit},
        {"$sort": {"timestamp": 1}}
    ]
    try:
        recent_convs = list(conversations.aggregate(pipeline))
        for conv in recent_convs:
            speaker_name = conv.get("speaker_name", users_collection.find_one({"user_id": conv["speaker_id"]})["name"])
            timestamp = conv["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            history.append({
                "speaker": speaker_name,
                "content": conv["content"],
                "timestamp": timestamp,
                "type": conv["type"],
                "source": conv.get("source", "human"),
                "raw_timestamp": conv["timestamp"],
                "conversation_id": conv["conversation_id"]
            })
        logger.info(f"Retrieved {len(history)} history entries")
        return history
    except Exception as e:
        logger.error(f"Failed to retrieve conversation history: {str(e)}")
        return []

def generate_personality_traits(user_id: str) -> dict:
    logger.info(f"Generating personality traits for user_id={user_id}")
    convs = list(conversations.find({"user_id": {"$elemMatch": {"$eq": user_id}}}).sort("timestamp", -1).limit(500))
    journals = list(journal_collection.find({"user_id": [user_id]}).sort("timestamp", -1).limit(500))
    data_text = "\n".join([c["content"] for c in convs] + [j["content"] for j in journals])[:1000]

    if not data_text:
        logger.warning(f"No data for user {user_id}")
        return {"core_traits": {}, "sub_traits": []}

    cached_traits = personalities_collection.find_one({"user_id": user_id})
    if cached_traits and "traits" in cached_traits:
        logger.info(f"Using cached traits for user {user_id}")
        return cached_traits["traits"]

    big_five_prompt = f"""
    Analyze this text from {users_collection.find_one({"user_id": user_id})["name"]}:
    {data_text}

    Return a JSON object with:
    - "core_traits": 5 traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) with scores (0-100) and one-sentence explanations.
    - "sub_traits": 3 unique traits with one-sentence descriptions.
    Ensure the response is concise to fit within 700 tokens.
    """

    for attempt in range(3):
        try:
            response = client2.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates personality traits."},
                    {"role": "user", "content": big_five_prompt}
                ],
                max_tokens=700,
                temperature=0.7
            )
            response_text = response.choices[0].message.content.strip()
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
            traits = json.loads(response_text)
            if "core_traits" in traits and "sub_traits" in traits:
                if isinstance(traits["core_traits"], list):
                    traits["core_traits"] = {t["trait"]: {"score": t["score"], "explanation": t["explanation"]} for t in traits["core_traits"]}
                logger.info(f"Generated traits for user {user_id}")
                break
        except Exception as e:
            logger.error(f"Trait generation attempt {attempt + 1} failed: {str(e)}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                traits = {
                    "core_traits": {
                        "Openness": {"score": 50, "explanation": "Neutral openness."},
                        "Conscientiousness": {"score": 50, "explanation": "Neutral conscientiousness."},
                        "Extraversion": {"score": 50, "explanation": "Neutral extraversion."},
                        "Agreeableness": {"score": 50, "explanation": "Neutral agreeableness."},
                        "Neuroticism": {"score": 50, "explanation": "Neutral neuroticism."}
                    },
                    "sub_traits": [
                        {"trait": "neutral", "description": "Shows balanced behavior."},
                        {"trait": "adaptable", "description": "Adapts to context."},
                        {"trait": "curious", "description": "Engages with data."}
                    ]
                }

    personalities_collection.update_one(
        {"user_id": user_id},
        {"$set": {"traits": traits}},
        upsert=True
    )
    return traits

import time

def get_greeting_and_tone(bot_role: str, target_id: str) -> tuple:
    logger.info(f"Generating greeting and tone for bot_role={bot_role}, target_id={target_id}")
    greeting_key = f"greeting_{target_id}_{bot_role}"
    cached_greeting = db.greetings.find_one({"key": greeting_key, "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=1)}})
    if cached_greeting:
        return cached_greeting["greeting"], cached_greeting["tone"]

    saved_greeting = saved_greetings.find_one(
        {"target_id": target_id, "bot_role": bot_role.lower()},
        sort=[("timestamp", -1)]
    )
    if saved_greeting:
        logger.debug(f"Using saved greeting: {saved_greeting['greeting']}")
        return saved_greeting["greeting"], "warm, youthful" if bot_role.lower() in ["daughter", "son"] else "nurturing, caring"

    default_greetings = {
        "daughter": ("Hey, Mom", "warm, youthful"),
        "son": ("Hey, Mom", "warm, youthful"),
        "mother": ("Hi, sweetie", "nurturing, caring"),
        "father": ("Hey, kid", "warm, supportive"),
        "sister": ("Yo, sis", "playful, casual"),
        "brother": ("Yo, bro", "playful, casual"),
        "wife": ("Hey, love", "affectionate, conversational"),
        "husband": ("Hey, hon", "affectionate, conversational"),
        "friend": ("Hey, what's good?", "casual, friendly")
    }
    greeting, tone = default_greetings.get(bot_role.lower(), ("Hey", "casual, friendly"))

    traits = generate_personality_traits(target_id)
    prompt = f"""
    You are generating a greeting for a {bot_role} with traits: {', '.join([f"{k}" for k in traits['core_traits'].keys()])}.
    Return a JSON object: {{"greeting": "short greeting (e.g., 'Hey, Mom')", "tone": "tone description (e.g., 'warm, youthful')"}}
    Ensure the response is valid JSON and nothing else.
    """
    for attempt in range(3):
        try:
            response = client2.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Return only valid JSON with 'greeting' and 'tone' keys."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.5
            )
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Raw greeting response: {response_text}")
            response_text = re.sub(r'^```json\s*|\s*```$', '', response_text, flags=re.MULTILINE).strip()
            result = json.loads(response_text)
            if "greeting" not in result or "tone" not in result:
                raise KeyError("Missing 'greeting' or 'tone' in response")
            greeting, tone = result["greeting"], result["tone"]
            break
        except Exception as e:
            logger.error(f"Greeting generation attempt {attempt + 1} failed: {str(e)}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.warning("Falling back to default greeting")
                break

    db.greetings.update_one(
        {"key": greeting_key},
        {"$set": {"greeting": greeting, "tone": tone, "timestamp": datetime.utcnow()}},
        upsert=True
    )
    return greeting, tone

def find_relevant_memories(speaker_id: str, user_id: str, user_input: str, speaker_name: str, max_memories: int = 5) -> List[dict]:
    global faiss_store
    logger.info(f"Finding memories: speaker={speaker_id}, target={user_id}, input='{user_input[:50]}...'")
    processed_input = preprocess_input(user_input)
    cache_key = f"input_{hash(processed_input)}"
    
    if cache_key in embedding_cache:
        input_embedding = embedding_cache[cache_key]
    else:
        input_embedding = embeddings.embed_query(processed_input)
        embedding_cache[cache_key] = input_embedding

    memories = []
    target_name = users_collection.find_one({"user_id": user_id})["name"]

    try:
        if faiss_store is None:
            logger.warning("FAISS store not initialized")
            return []

        results = faiss_store.similarity_search_with_score(processed_input, k=max_memories * 3)
        logger.info(f"Found {len(results)} potential memories from FAISS")

        for doc, score in results:
            metadata = doc.metadata
            item_id = metadata.get("item_id")
            item_type = metadata.get("item_type")
            if not item_id or not item_type:
                logger.warning(f"Invalid metadata: {metadata}")
                continue

            collection = conversations if item_type == "conversation" else journal_collection
            id_field = "conversation_id" if item_type == "conversation" else "entry_id"
            query = {
                id_field: item_id,
                "user_id": [user_id] if item_type == "journal" else {"$in": [[speaker_id, user_id], [user_id, speaker_id]]}
            }
            db_doc = collection.find_one(query)
            if not db_doc:
                logger.warning(f"No document found for item_id: {item_id}, item_type: {item_type}, query: {query}")
                embeddings_collection.delete_one({"item_id": item_id, "item_type": item_type})
                continue

            required_fields = ["content", "timestamp"]
            if item_type == "conversation":
                required_fields.append("speaker_name")
            if not all(f in db_doc for f in required_fields):
                logger.warning(f"Missing required fields in document: {item_id}, fields: {db_doc.keys()}")
                continue

            if item_type == "journal":
                db_doc["speaker_name"] = target_name

            adjusted_score = 1.0 - score
            if item_type == "journal" and user_id in metadata["user_id"]:
                adjusted_score += 0.9
            elif metadata.get("speaker_id") == speaker_id or metadata.get("target_id") == user_id:
                adjusted_score += 0.7
            if speaker_name.lower() in str(db_doc.get("content", "")).lower() or target_name.lower() in str(db_doc.get("content", "")).lower():
                adjusted_score += 0.3
            days_old = (datetime.utcnow() - metadata["timestamp"]).days
            temporal_weight = 1 / (1 + np.log1p(max(days_old, 1) / 30))
            adjusted_score *= temporal_weight

            if adjusted_score < 0.3:
                logger.debug(f"Skipping memory {item_id} with low score: {adjusted_score}")
                continue

            memory = {
                "type": item_type,
                "content": db_doc["content"],
                "timestamp": db_doc["timestamp"],
                "score": float(adjusted_score),
                "user_id": metadata["user_id"],
                "speaker_id": metadata.get("speaker_id", user_id if item_type == "journal" else None),
                "speaker_name": db_doc.get("speaker_name", target_name),
                "target_id": metadata.get("target_id", None),
                "target_name": metadata.get("target_name", None)
            }
            memories.append(memory)
            logger.debug(f"Added memory: item_id={item_id}, type={item_type}, score={adjusted_score}, content={db_doc['content'][:50]}...")

        return sorted(memories, key=lambda x: x["score"], reverse=True)[:max_memories]
    except Exception as e:
        logger.error(f"FAISS search failed: {str(e)}")
        return []
    
def should_include_memories(user_input: str, speaker_id: str, user_id: str) -> tuple:
    logger.info(f"Checking memory inclusion for input='{user_input[:50]}...'")
    memories = find_relevant_memories(speaker_id, user_id, user_input,
        users_collection.find_one({"user_id": speaker_id})["name"], max_memories=10)

    relevant_memories = []
    if memories:
        processed_input = preprocess_input(user_input)
        input_embedding = embeddings.embed_query(processed_input)
        for m in memories:
            memory_embedding = embeddings.embed_query(m["content"])
            similarity = np.dot(input_embedding, memory_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(memory_embedding))
            if similarity >= 0.5:
                relevant_memories.append(m)
    return bool(relevant_memories), relevant_memories[:3]

def initialize_bot(speaker_id: str, target_id: str, bot_role: str, user_input: str) -> tuple:
    logger.info(f"Initializing bot: speaker={speaker_id}, target={target_id}, bot_role={bot_role}")
    speaker = users_collection.find_one({"user_id": speaker_id})
    target = users_collection.find_one({"user_id": target_id})
    if not speaker or not target:
        raise ValueError("Invalid speaker or target ID")

    traits = generate_personality_traits(target_id)
    recent_history = get_recent_conversation_history(speaker_id, target_id)
    history_text = "\n".join([f"{msg['speaker']}: {msg['content']}" for msg in recent_history]) or "No recent conversation history."
    use_greeting = not recent_history or (datetime.utcnow() - recent_history[-1]["raw_timestamp"]).total_seconds() / 60 > 30
    greeting, tone = get_greeting_and_tone(bot_role, target_id)

    include_memories, memories = should_include_memories(user_input, speaker_id, target_id)
    memories_text = "No relevant memories."
    if include_memories and memories:
        valid_memories = [m for m in memories if all(key in m for key in ["content", "type", "timestamp", "speaker_name"])]
        if valid_memories:
            memories_text = "\n".join([
                f"- {m['content']} ({m['type']}, {m['timestamp'].strftime('%Y-%m-%d')}, said by {m['speaker_name']})"
                for m in valid_memories
            ])

    if include_memories:
        prompt = f"""
        You are {target['name']}, responding as an AI Twin to {speaker['name']}, you are his/her {bot_role}.
        Generate a short 2-3 sentence reply that:
        - Uses a {tone} tone, appropriate for your relationship with {speaker['name']}.
        - Reflects your personality: {', '.join([f"{k} ({v['explanation']})" for k, v in list(traits['core_traits'].items())[:3]])}.
        - Uses this recent context:
        {history_text}
        - If relevant to '{user_input}', weave in one or two of these memories naturally, clearly attributing them to their speaker:
        {memories_text}
        - Prioritize recent and highly relevant memories; stick to these details strictly and do not invent details.
        - {'Starts with "' + greeting + '" if no recent messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
        - Keeps responses short, casual, and personalized.
        Input: {user_input}
        """
    else:
        prompt = f"""
        You are {target['name']}, responding as an AI Twin to {speaker['name']}, his/her {bot_role}.
        Generate a short 2-3 sentence reply that:
        - Uses a {tone} tone, appropriate for your relationship with {speaker['name']}.
        - Reflects your personality: {', '.join([f"{k} ({v['explanation']})" for k, v in list(traits['core_traits'].items())[:3]])}.
        - Uses this recent conversation history for context:
        {history_text}
        - Focuses on the current input without referencing past memories unless explicitly relevant; stick to these details strictly and do not invent details.
        - {'Starts with "' + greeting + '" if no recent messages or time gap > 30 minutes.' if use_greeting else 'Do not start with a greeting.'}
        - Keeps responses short, casual, and personalized.
        Input: {user_input}
        """
    return prompt, greeting, use_greeting

def generate_response(prompt: str, user_input: str, greeting: str, use_greeting: bool) -> str:
    logger.info(f"Generating response for input='{user_input[:50]}...'")
    try:
        response = client2.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI Twin responding in a personalized, casual manner."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.6
        )
        response_text = response.choices[0].message.content.strip()
        if len(response_text.split()) >= 4 and (use_greeting and response_text.lower().startswith(greeting.lower()) or not use_greeting):
            sentences = response_text.split('. ')[:3]
            response_text = '. '.join([s for s in sentences if s]).strip()
            if response_text and not response_text.endswith('.'):
                response_text += '.'
            return response_text
    except Exception as e:
        logger.error(f"OpenAI failed: {str(e)}")
        errors_collection.insert_one({"error": str(e), "input": user_input, "timestamp": datetime.utcnow()})
    return f"{greeting}, sounds cool! What's up?" if use_greeting else "Sounds cool! What's up?"


def clear_and_repopulate():
    logger.info("Clearing and repopulating database")
    clear_database()
    embedding_cache.clear()
    populate_users()
    populate_conversations()
    populate_journals()
    verify_data()
    # Verify embeddings before initializing FAISS
    embeddings_count = embeddings_collection.count_documents({})
    if embeddings_count == 0:
        logger.info("No embeddings found, initializing empty FAISS store")
        faiss_store = FAISS.from_texts(["empty"], embeddings)
        return



@app.post("/send_message", response_model=MessageResponse)
async def send_message(request: MessageRequest, x_api_key: str = Header(...)):
    global faiss_store
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    embedding_cache.clear()
    logger.info("Cleared embedding cache")

    logger.info(f"Processing message: speaker={request.speaker_id}, target={request.target_id}, role={request.bot_role}")
    try:
        # Store user input
        processed_input = preprocess_input(request.user_input)
        user_conv_id = str(uuid.uuid4())
        conv_doc = {
            "conversation_id": user_conv_id,
            "user_id": [request.speaker_id, request.target_id],
            "speaker_id": request.speaker_id,
            "speaker_name": users_collection.find_one({"user_id": request.speaker_id})["name"],
            "target_id": request.target_id,
            "target_name": users_collection.find_one({"user_id": request.target_id})["name"],
            "content": request.user_input,
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow()
        }
        conversations.insert_one(conv_doc)
        embedding = embeddings.embed_query(processed_input)
        embeddings_collection.insert_one({
            "item_id": user_conv_id,
            "item_type": "conversation",
            "user_id": [request.speaker_id, request.target_id],
            "speaker_id": request.speaker_id,
            "target_id": request.target_id,
            "speaker_name": conv_doc["speaker_name"],
            "target_name": conv_doc["target_name"],
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "content": request.user_input  # Store content for FAISS
        })

        # Add to FAISS store
        try:
            doc = Document(
                page_content=request.user_input,
                metadata={
                    "item_id": user_conv_id,
                    "item_type": "conversation",
                    "user_id": [request.speaker_id, request.target_id],
                    "speaker_id": request.speaker_id,
                    "target_id": request.target_id,
                    "speaker_name": conv_doc["speaker_name"],
                    "target_name": conv_doc["target_name"],
                    "timestamp": datetime.utcnow()
                }
            )
            faiss_store.add_documents([doc])
            logger.info(f"Added user input to FAISS store: {user_conv_id}")
        except Exception as e:
            logger.error(f"Failed to add to FAISS store: {str(e)}")

        # Generate response
        prompt, greeting, use_greeting = initialize_bot(
            request.speaker_id, request.target_id, request.bot_role, request.user_input
        )
        response_text = generate_response(prompt, request.user_input, greeting, use_greeting)

        # Store response
        bot_conv_id = str(uuid.uuid4())
        processed_response = preprocess_input(response_text)
        conv_doc = {
            "conversation_id": bot_conv_id,
            "user_id": [request.speaker_id, request.target_id],
            "speaker_id": request.target_id,
            "speaker_name": users_collection.find_one({"user_id": request.target_id})["name"],
            "target_id": request.speaker_id,
            "target_name": users_collection.find_one({"user_id": request.speaker_id})["name"],
            "content": response_text,
            "type": "response",
            "source": "ai_twin",
            "timestamp": datetime.utcnow()
        }
        conversations.insert_one(conv_doc)
        embedding = embeddings.embed_query(processed_response)
        embeddings_collection.insert_one({
            "item_id": bot_conv_id,
            "item_type": "conversation",
            "user_id": [request.speaker_id, request.target_id],
            "speaker_id": request.target_id,
            "target_id": request.speaker_id,
            "speaker_name": conv_doc["speaker_name"],
            "target_name": conv_doc["target_name"],
            "embedding": embedding,
            "timestamp": datetime.utcnow(),
            "content": response_text  # Store content for FAISS
        })

        # Add response to FAISS store
        try:
            doc = Document(
                page_content=response_text,
                metadata={
                    "item_id": bot_conv_id,
                    "item_type": "conversation",
                    "user_id": [request.speaker_id, request.target_id],
                    "speaker_id": request.target_id,
                    "target_id": request.speaker_id,
                    "speaker_name": conv_doc["speaker_name"],
                    "target_name": conv_doc["target_name"],
                    "timestamp": datetime.utcnow()
                }
            )
            faiss_store.add_documents([doc])
            logger.info(f"Added bot response to FAISS store: {bot_conv_id}")
        except Exception as e:
            logger.error(f"Failed to add to FAISS store: {str(e)}")

        return MessageResponse(response=response_text)
    except Exception as e:
        logger.error(f"Interaction failed: {str(e)}")
        errors_collection.insert_one({"error": str(e), "input": request.user_input, "timestamp": datetime.utcnow()})
        return MessageResponse(response="", error=str(e))


def clear_and_repopulate():
    global faiss_store  # Declare global at the start
    logger.info("Clearing and repopulating database")
    clear_database()
    faiss_store = None
    embedding_cache.clear()
    populate_users()
    populate_conversations()
    populate_journals()
    verify_data()
    # Verify embeddings before initializing FAISS
    embeddings_count = embeddings_collection.count_documents({})
    if embeddings_count == 0:
        logger.info("No embeddings found, initializing empty FAISS store")
        faiss_store = FAISS.from_texts(["empty"], embeddings)
        return


def process_new_entry(item_id: str, item_type: str, content: str, user_id: list, speaker_id: Optional[str] = None, 
                     speaker_name: Optional[str] = None, target_id: Optional[str] = None, target_name: Optional[str] = None):
    global faiss_store
    logger.info(f"Processing embedding for {item_type} with item_id={item_id}")
    try:
        processed_content = preprocess_input(content)
        embedding = embeddings.embed_query(processed_content)
        
        embedding_doc = {
            "item_id": item_id,
            "item_type": item_type,
            "user_id": user_id,
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.utcnow()
        }
        if item_type == "conversation":
            embedding_doc.update({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "target_id": target_id,
                "target_name": target_name
            })
        
        embeddings_collection.insert_one(embedding_doc)
        logger.info(f"Inserted embedding for {item_type} {item_id}")
        
        if faiss_store is None:
            logger.warning("FAISS store is None, initializing empty store")
            faiss_store = FAISS.from_texts(["empty"], embeddings)
        
        metadata = {
            "item_id": item_id,
            "item_type": item_type,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        }
        if item_type == "conversation":
            metadata.update({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "target_id": target_id,
                "target_name": target_name
            })
        doc = Document(page_content=content, metadata=metadata)
        faiss_store.add_documents([doc])
        logger.info(f"Added {item_type} to FAISS store: {item_id}")
    except Exception as e:
        logger.error(f"Failed to process embedding for {item_type} {item_id}: {str(e)}")
        errors_collection.insert_one({"error": str(e), "item_id": item_id, "item_type": item_type, "timestamp": datetime.utcnow()})

import threading
from pymongo import MongoClient

def watch_collections():
    logger.info("Starting change stream watchers for conversations and journal_entries")
    while True:
        try:
            with conversations.watch([{"$match": {"operationType": "insert"}}]) as stream:
                for change in stream:
                    doc = change["fullDocument"]
                    if doc.get("type") == "user_input" and doc.get("source") == "human":
                        logger.info(f"Detected new conversation: {doc['conversation_id']}")
                        process_new_entry(
                            item_id=doc["conversation_id"],
                            item_type="conversation",
                            content=doc["content"],
                            user_id=doc["user_id"],
                            speaker_id=doc["speaker_id"],
                            speaker_name=doc["speaker_name"],
                            target_id=doc["target_id"],
                            target_name=doc["target_name"]
                        )
        except Exception as e:
            logger.error(f"Conversation change stream failed: {str(e)}")
            errors_collection.insert_one({"error": str(e), "collection": "conversations", "timestamp": datetime.utcnow()})
            time.sleep(5)  # Wait before retrying
        
        try:
            with journal_collection.watch([{"$match": {"operationType": "insert"}}]) as stream:
                for change in stream:
                    doc = change["fullDocument"]
                    logger.info(f"Detected new journal entry: {doc['entry_id']}")
                    process_new_entry(
                        item_id=doc["entry_id"],
                        item_type="journal",
                        content=doc["content"],
                        user_id=doc["user_id"]
                    )
        except Exception as e:
            logger.error(f"Journal change stream failed: {str(e)}")
            errors_collection.insert_one({"error": str(e), "collection": "journal_entries", "timestamp": datetime.utcnow()})
            time.sleep(5)  # Wait before retrying

# Start change stream watchers in a background thread
def start_change_stream_watcher():
    logger.info("Attempting to start change stream watcher thread")
    watcher_thread = threading.Thread(target=watch_collections, daemon=True)
    watcher_thread.start()
    logger.info("Change stream watcher thread started successfully")

if __name__ == "__main__":
    clear_and_repopulate()
    start_change_stream_watcher()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
