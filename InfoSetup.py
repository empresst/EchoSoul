import os
from pymongo import MongoClient
from datetime import datetime
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    logger.error("MONGODB_URI not found in .env file")
    raise ValueError("MONGODB_URI not found")

# MongoDB setup
logger.info("Connecting to MongoDB Atlas")
client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["user_bots"]
users_collection = db["users"]
conversations = db["conversations"]
journal_collection = db["journal_entries"]
embeddings_collection = db["embeddings"]
saved_greetings = db["saved_greetings"]

# Load SentenceTransformer for embeddings
logger.info("Loading SentenceTransformer model")
retriever = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# Demo conversations
additional_conversations = [
    {
        "conversation_id": "conv14",
        "user_id": ["user1", "user2"],
        "speaker_id": "user1",
        "speaker_name": "Alice",
        "target_id": "user2",
        "target_name": "Bob",
        "content": "Bob, let's plan a movie night this weekend!",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 13, 14, 0, 0)
    },
    {
        "conversation_id": "conv15",
        "user_id": ["user1", "user2"],
        "speaker_id": "user2",
        "speaker_name": "Bob",
        "target_id": "user1",
        "target_name": "Alice",
        "content": "Sweet, I'm in! Action or comedy?",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 13, 14, 5, 0)
    },
    {
        "conversation_id": "conv16",
        "user_id": ["user1", "user2"],
        "speaker_id": "user1",
        "speaker_name": "Alice",
        "target_id": "user2",
        "target_name": "Bob",
        "content": "Let's go with comedy. Got any favorites?",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 13, 14, 10, 0)
    },
    {
        "conversation_id": "conv17",
        "user_id": ["user3", "user4"],
        "speaker_id": "user3",
        "speaker_name": "Charlie",
        "target_id": "user4",
        "target_name": "Diana",
        "content": "Diana, the kids are planning a movie night. Should we join?",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 13, 15, 0, 0)
    },
    {
        "conversation_id": "conv18",
        "user_id": ["user1", "user2"],
        "speaker_id": "user2",
        "speaker_name": "Bob",
        "target_id": "user1",
        "target_name": "Alice",
        "content": "I’m so into comedy movies, but I can’t stand horror—too creepy!",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 14, 10, 0, 0)
    },
    {
        "conversation_id": "conv19",
        "user_id": ["user1", "user2"],
        "speaker_id": "user1",
        "speaker_name": "Alice",
        "target_id": "user2",
        "target_name": "Bob",
        "content": "Horror’s not my thing either, but I love thrillers. Also, congrats on graduating last month!",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 14, 10, 5, 0)
    },
    {
        "conversation_id": "conv20",
        "user_id": ["user3", "user4"],
        "speaker_id": "user3",
        "speaker_name": "Charlie",
        "target_id": "user4",
        "target_name": "Diana",
        "content": "Got promoted to manager last week! Wanna celebrate with a fancy dinner?",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 15, 12, 0, 0)
    },
    {
        "conversation_id": "conv21",
        "user_id": ["user3", "user4"],
        "speaker_id": "user4",
        "speaker_name": "Diana",
        "target_id": "user3",
        "target_name": "Charlie",
        "content": "That’s amazing, Charlie! I’m not a fan of spicy food, but let’s pick a nice Italian place.",
        "type": "user_input",
        "source": "human",
        "timestamp": datetime(2025, 5, 15, 12, 5, 0)
    },
]

# Demo journal entries
additional_journals = {
    "user1": [
        {"entry_id": "journal12", "content": "I love comedy and thriller movie!", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
        {"entry_id": "journal13", "content": "Feeling a bit nervous about my presentation tomorrow.", "timestamp": datetime(2025, 5, 14, 9, 0, 0)},
        {"entry_id": "journal16", "content": "Bob’s graduation was such a proud moment. I hate horror movies, they give me nightmares!", "timestamp": datetime(2025, 5, 15, 8, 0, 0)},
    ],
    "user2": [
        {"entry_id": "journal14", "content": "Alice’s movie night idea is awesome. Gotta pick a good comedy.", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
        {"entry_id": "journal17", "content": "Graduating felt unreal! I’m sticking to comedy movies to celebrate.", "timestamp": datetime(2025, 5, 14, 9, 0, 0)},
    ],
    "user3": [
        {"entry_id": "journal15", "content": "The kids are growing up so fast. Movie night sounds like a great family plan.", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
        {"entry_id": "journal18", "content": "My promotion to manager is a big step. I dislike crowded places, so I’m avoiding busy restaurants.", "timestamp": datetime(2025, 5, 16, 9, 0, 0)},
    ],
    "user4": [
        {"entry_id": "journal19", "content": "Our 10th wedding anniversary was magical. I love Italian food but dislike spicy dishes.", "timestamp": datetime(2025, 5, 16, 9, 0, 0)},
    ],
}

def add_demo_data():
    logger.info("Starting add_demo_data")
    
    # Ensure demo users exist
    demo_users = [
        {"user_id": "user1", "name": "Alice"},
        {"user_id": "user2", "name": "Bob"},
        {"user_id": "user3", "name": "Charlie"},
        {"user_id": "user4", "name": "Diana"}
    ]
    for user in demo_users:
        if not users_collection.find_one({"user_id": user["user_id"]}):
            users_collection.insert_one(user)
            logger.debug(f"Inserted user: {user['user_id']}")

    # Insert additional conversations
    for conv in additional_conversations:
        if conversations.find_one({"conversation_id": conv["conversation_id"]}):
            logger.debug(f"Skipping existing conversation: {conv['conversation_id']}")
            continue
        conversations.insert_one(conv)
        logger.debug(f"Inserted conversation: {conv['conversation_id']}")

    # Insert additional journal entries and generate embeddings
    for user_id, entries in additional_journals.items():
        user = users_collection.find_one({"user_id": user_id})
        if not user:
            logger.error(f"User {user_id} not found")
            continue
        for entry in entries:
            if journal_collection.find_one({"entry_id": entry["entry_id"]}):
                logger.debug(f"Skipping existing journal: {entry['entry_id']}")
                continue
            journal_doc = {
                "entry_id": entry["entry_id"],
                "user_id": [user_id],
                "speaker_id": user_id,
                "speaker_name": user["name"],
                "content": entry["content"],
                "timestamp": entry["timestamp"]
            }
            journal_collection.insert_one(journal_doc)
            logger.debug(f"Inserted journal: {entry['entry_id']} for user {user_id}")

            # Generate and store embedding
            if embeddings_collection.find_one({"item_id": entry["entry_id"], "item_type": "journal"}):
                logger.debug(f"Skipping existing embedding for journal: {entry['entry_id']}")
                continue
            embedding = retriever.encode(entry["content"], normalize_embeddings=True).tolist()
            embedding_doc = {
                "item_id": entry["entry_id"],
                "item_type": "journal",
                "user_id": [user_id],
                "speaker_id": user_id,
                "speaker_name": user["name"],
                "embedding": embedding,
                "timestamp": entry["timestamp"]
            }
            embeddings_collection.insert_one(embedding_doc)
            logger.debug(f"Inserted embedding for journal: {entry['entry_id']}")

    logger.info(f"Conversations count: {conversations.count_documents({})}")
    logger.info(f"Journal entries count: {journal_collection.count_documents({})}")
    logger.info(f"Embeddings count: {embeddings_collection.count_documents({})}")

if __name__ == "__main__":
    add_demo_data()
