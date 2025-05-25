# import os
# import uuid
# import logging
# from datetime import datetime, timedelta
# from pymongo import MongoClient
# from pymongo.errors import PyMongoError
# from dotenv import load_dotenv
# from langchain_openai import OpenAIEmbeddings

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MONGODB_URI = os.getenv("MONGODB_URI")
# if not OPENAI_API_KEY or not MONGODB_URI:
#     raise ValueError("Missing OPENAI_API_KEY or MONGODB_URI")

# # MongoDB setup
# client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
# db = client["LF"]
# users_collection = db["users"]
# conversations_collection = db["conversations"]
# journal_collection = db["journal_entries"]
# embeddings_collection = db["embeddings"]

# # OpenAI embeddings
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# def clear_database():
#     logger.info("Clearing database collections")
#     users_collection.delete_many({})
#     conversations_collection.delete_many({})
#     journal_collection.delete_many({})
#     embeddings_collection.delete_many({})

#     logger.info("Database cleared")
# def populate_users():
#     """Add 4 users, skipping duplicates."""
#     users = [
#         {"user_id": "user1", "name": "Alice"},
#         {"user_id": "user2", "name": "Bob"},
#         {"user_id": "user3", "name": "Charlie"},
#         {"user_id": "user4", "name": "Diana"}
#     ]
#     try:
#         for user in users:
#             if not users_collection.find_one({"user_id": user["user_id"]}):
#                 users_collection.insert_one(user)
#                 logger.info(f"Inserted user: {user['user_id']}")
#     except PyMongoError as e:
#         logger.error(f"Failed to populate users: {str(e)}")
#         raise

# def batch_embed_texts(texts):
#     """Generate embeddings for texts."""
#     try:
#         return embeddings.embed_documents(texts)
#     except Exception as e:
#         logger.warning(f"Failed to embed texts: {str(e)}")
#         return [None] * len(texts)

# def populate_conversations():
#     """Add 4 conversation messages, skipping duplicates."""
#     conversations = [
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_id": ["user1", "user2"],
#             "speaker_id": "user1",
#             "speaker_name": "Alice",
#             "target_id": "user2",
#             "target_name": "Bob",
#             "content": "Hey Bob, ready for the project?",
#             "type": "user_input",
#             "source": "human",
#             "timestamp": datetime.utcnow() - timedelta(days=1)
#         },
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_id": ["user2", "user1"],
#             "speaker_id": "user2",
#             "speaker_name": "Bob",
#             "target_id": "user1",
#             "target_name": "Alice",
#             "content": "Yeah, let's do this!",
#             "type": "user_input",
#             "source": "human",
#             "timestamp": datetime.utcnow() - timedelta(days=1, hours=1)
#         },
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_id": ["user3", "user4"],
#             "speaker_id": "user3",
#             "speaker_name": "Charlie",
#             "target_id": "user4",
#             "target_name": "Diana",
#             "content": "Diana, got any weekend plans?",
#             "type": "user_input",
#             "source": "human",
#             "timestamp": datetime.utcnow() - timedelta(days=2)
#         },
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_id": ["user4", "user3"],
#             "speaker_id": "user4",
#             "speaker_name": "Diana",
#             "target_id": "user3",
#             "target_name": "Charlie",
#             "content": "Just chilling, you?",
#             "type": "user_input",
#             "source": "human",
#             "timestamp": datetime.utcnow() - timedelta(days=2, hours=1)
#         }
#     ]
#     try:
#         for conv in conversations:
#             if not conversations_collection.find_one({"conversation_id": conv["conversation_id"]}):
#                 conversations_collection.insert_one(conv)
#         contents = [conv["content"] for conv in conversations]
#         embedding_results = batch_embed_texts(contents)
#         embedding_docs = [
#             {
#                 "item_id": conv["conversation_id"],
#                 "item_type": "conversation",
#                 "user_id": conv["user_id"],
#                 "content": conv["content"],
#                 "embedding": embedding,
#                 "timestamp": conv["timestamp"]
#             }
#             for conv, embedding in zip(conversations, embedding_results)
#             if embedding and not embeddings_collection.find_one({"item_id": conv["conversation_id"], "item_type": "conversation"})
#         ]
#         if embedding_docs:
#             embeddings_collection.insert_many(embedding_docs)
#         logger.info("Conversations populated")
#     except PyMongoError as e:
#         logger.error(f"Failed to populate conversations: {str(e)}")
#         raise

# def populate_journals():
#     """Add 4 journal entries, skipping duplicates."""
#     journals = [
#         {
#             "entry_id": str(uuid.uuid4()),
#             "user_id": ["user1"],
#             "content": "painting is my hobby!",
#             "timestamp": datetime.utcnow() - timedelta(days=1)
#         },
#         {
#             "entry_id": str(uuid.uuid4()),
#             "user_id": ["user2"],
#             "content": "Hiking is my hobby!",
#             "timestamp": datetime.utcnow() - timedelta(days=1, hours=2)
#         },
#         {
#             "entry_id": str(uuid.uuid4()),
#             "user_id": ["user3"],
#             "content": "cooking is my hobby.",
#             "timestamp": datetime.utcnow() - timedelta(days=2)
#         },
#         {
#             "entry_id": str(uuid.uuid4()),
#             "user_id": ["user4"],
#             "content": "travelling is my hobby.",
#             "timestamp": datetime.utcnow() - timedelta(days=2, hours=2)
#         }
#     ]
#     try:
#         for journal in journals:
#             if not journal_collection.find_one({"entry_id": journal["entry_id"]}):
#                 journal_collection.insert_one(journal)
#         contents = [journal["content"] for journal in journals]
#         embedding_results = batch_embed_texts(contents)
#         embedding_docs = [
#             {
#                 "item_id": journal["entry_id"],
#                 "item_type": "journal",
#                 "user_id": journal["user_id"],
#                 "content": journal["content"],
#                 "embedding": embedding,
#                 "timestamp": journal["timestamp"]
#             }
#             for journal, embedding in zip(journals, embedding_results)
#             if embedding and not embeddings_collection.find_one({"item_id": journal["entry_id"], "item_type": "journal"})
#         ]
#         if embedding_docs:
#             embeddings_collection.insert_many(embedding_docs)
#         logger.info("Journals populated")
#     except PyMongoError as e:
#         logger.error(f"Failed to populate journals: {str(e)}")
#         raise

# def verify_data():
#     """Verify record counts in collections."""
#     try:
#         counts = {
#             "Users": users_collection.count_documents({}),
#             "Conversations": conversations_collection.count_documents({}),
#             "Journals": journal_collection.count_documents({}),
#             "Embeddings": embeddings_collection.count_documents({})
#         }
#         logger.info(f"Database contents: {counts}")
#     except PyMongoError as e:
#         logger.error(f"Failed to verify data: {str(e)}")
#         raise

# def main():
#     try:
#         clear_database()
#         populate_users()
#         populate_conversations()
#         populate_journals()
#         verify_data()
#         logger.info("Database population completed")
#     except Exception as e:
#         logger.error(f"Database population failed: {str(e)}")
#         raise
#     finally:
#         client.close()

# if __name__ == "__main__":
#     main()



import os
import uuid
import logging
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
if not OPENAI_API_KEY or not MONGODB_URI:
    raise ValueError("Missing OPENAI_API_KEY or MONGODB_URI")

# MongoDB setup
client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
db = client["LF"]
users_collection = db["users"]
conversations_collection = db["conversations"]
journal_collection = db["journal_entries"]
embeddings_collection = db["embeddings"]

# OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

def clear_database():
    logger.info("Clearing database collections")
    users_collection.delete_many({})
    conversations_collection.delete_many({})
    journal_collection.delete_many({})
    embeddings_collection.delete_many({})
    logger.info("Database cleared")

def populate_users():
    """Add 4 users, skipping duplicates."""
    users = [
        {"user_id": "user1", "name": "Nipa"},
        {"user_id": "user2", "name": "Nick"},
        {"user_id": "user3", "name": "Arif"},
        {"user_id": "user4", "name": "Diana"}
    ]
    try:
        for user in users:
            if not users_collection.find_one({"user_id": user["user_id"]}):
                users_collection.insert_one(user)
                logger.info(f"Inserted user: {user['user_id']}")
    except PyMongoError as e:
        logger.error(f"Failed to populate users: {str(e)}")
        raise

def batch_embed_texts(texts):
    """Generate embeddings for texts."""
    try:
        return embeddings.embed_documents(texts)
    except Exception as e:
        logger.warning(f"Failed to embed texts: {str(e)}")
        return [None] * len(texts)

def populate_conversations():
    """Add 6 conversation messages, skipping duplicates."""
    conversations = [
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user1", "user2"],
            "speaker_id": "user1",
            "speaker_name": "Nipa",
            "target_id": "user2",
            "target_name": "Nick",
            "content": "Hey nick, ready for the project?",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=1)
        },
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user2", "user1"],
            "speaker_id": "user2",
            "speaker_name": "Nick",
            "target_id": "user1",
            "target_name": "Nipa",
            "content": "Yeah, let's do this!",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=1, hours=1)
        },
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user3", "user4"],
            "speaker_id": "user3",
            "speaker_name": "Arif",
            "target_id": "user4",
            "target_name": "Diana",
            "content": "Diana, got any weekend plans?",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=2)
        },
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user4", "user3"],
            "speaker_id": "user4",
            "speaker_name": "Diana",
            "target_id": "user3",
            "target_name": "Arif",
            "content": "Just chilling, you?",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=2, hours=1)
        },
        # New conversation entries
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user1", "user3"],
            "speaker_id": "user1",
            "speaker_name": "Nipa",
            "target_id": "user3",
            "target_name": "Arif",
            "content": "Dad, I want to go to disney",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=0, hours=12)
        },
        {
            "conversation_id": str(uuid.uuid4()),
            "user_id": ["user4", "user2"],
            "speaker_id": "user4",
            "speaker_name": "Diana",
            "target_id": "user2",
            "target_name": "Nick",
            "content": "Nick, have you tried the new coffee shop yet?",
            "type": "user_input",
            "source": "human",
            "timestamp": datetime.utcnow() - timedelta(days=0, hours=10)
        }
    ]
    try:
        for conv in conversations:
            if not conversations_collection.find_one({"conversation_id": conv["conversation_id"]}):
                conversations_collection.insert_one(conv)
        contents = [conv["content"] for conv in conversations]
        embedding_results = batch_embed_texts(contents)
        embedding_docs = [
            {
                "item_id": conv["conversation_id"],
                "item_type": "conversation",
                "user_id": conv["user_id"],
                "content": conv["content"],
                "embedding": embedding,
                "timestamp": conv["timestamp"]
            }
            for conv, embedding in zip(conversations, embedding_results)
            if embedding and not embeddings_collection.find_one({"item_id": conv["conversation_id"], "item_type": "conversation"})
        ]
        if embedding_docs:
            embeddings_collection.insert_many(embedding_docs)
        logger.info("Conversations populated")
    except PyMongoError as e:
        logger.error(f"Failed to populate conversations: {str(e)}")
        raise

def populate_journals():
    """Add 6 journal entries, skipping duplicates."""
    journals = [
        # {
        #     "entry_id": str(uuid.uuid4()),
        #     "user_id": ["user1"],
        #     "content": "Painting is my hobby!",
        #     "timestamp": datetime.utcnow() - timedelta(days=1)
        # },
        # {
        #     "entry_id": str(uuid.uuid4()),
        #     "user_id": ["user2"],
        #     "content": "Hiking is my hobby!",
        #     "timestamp": datetime.utcnow() - timedelta(days=1, hours=2)
        # },
        # {
        #     "entry_id": str(uuid.uuid4()),
        #     "user_id": ["user3"],
        #     "content": "Cooking is my hobby.",
        #     "timestamp": datetime.utcnow() - timedelta(days=2)
        # },
        # {
        #     "entry_id": str(uuid.uuid4()),
        #     "user_id": ["user4"],
        #     "content": "Travelling is my hobby.",
        #     "timestamp": datetime.utcnow() - timedelta(days=2, hours=2)
        # },
        # # New journal entries
        # {
        #     "entry_id": str(uuid.uuid4()),
        #     "user_id": ["user1"],
        #     "content": "I want to be an engineer!",
        #     "timestamp": datetime.utcnow() - timedelta(days=0, hours=8)
        # },
        {
            "entry_id": str(uuid.uuid4()),
            "user_id": ["user1"],
            "content": "I am in love with Jack",
            "timestamp": datetime.utcnow() - timedelta(days=0, hours=6)
        }
    ]
    try:
        for journal in journals:
            if not journal_collection.find_one({"entry_id": journal["entry_id"]}):
                journal_collection.insert_one(journal)
        contents = [journal["content"] for journal in journals]
        embedding_results = batch_embed_texts(contents)
        embedding_docs = [
            {
                "item_id": journal["entry_id"],
                "item_type": "journal",
                "user_id": journal["user_id"],
                "content": journal["content"],
                "embedding": embedding,
                "timestamp": journal["timestamp"]
            }
            for journal, embedding in zip(journals, embedding_results)
            if embedding and not embeddings_collection.find_one({"item_id": journal["entry_id"], "item_type": "journal"})
        ]
        if embedding_docs:
            embeddings_collection.insert_many(embedding_docs)
        logger.info("Journals populated")
    except PyMongoError as e:
        logger.error(f"Failed to populate journals: {str(e)}")
        raise

def verify_data():
    """Verify record counts in collections."""
    try:
        counts = {
            "Users": users_collection.count_documents({}),
            "Conversations": conversations_collection.count_documents({}),
            "Journals": journal_collection.count_documents({}),
            "Embeddings": embeddings_collection.count_documents({})
        }
        logger.info(f"Database contents: {counts}")
    except PyMongoError as e:
        logger.error(f"Failed to verify data: {str(e)}")
        raise

def main():
    try:
        # clear_database()
        # populate_users()
        # populate_conversations()
        populate_journals()
        verify_data()
        logger.info("Database population completed")
    except Exception as e:
        logger.error(f"Database population failed: {str(e)}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    main()