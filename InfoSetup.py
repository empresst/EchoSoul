# import os
# import uuid
# from datetime import datetime
# import logging
# from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from dotenv import load_dotenv

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("setup_chatbot_data")

# # Load environment variables
# load_dotenv()

# MONGODB_URI = os.getenv("MONGODB_URI")
# client = MongoClient(MONGODB_URI, maxPoolSize=50, serverSelectionTimeoutMS=5000)
# db = client["user_bots"]
# users_collection = db["users"]
# conversations = db["conversations"]
# journal_collection = db["journal_entries"]
# embeddings_collection = db["embeddings"]
# saved_greetings = db["saved_greetings"]
# greetings_cache = db["greetings"]

# retriever = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# def quantize_embedding(embedding: np.ndarray) -> np.ndarray:
#     return (embedding / np.linalg.norm(embedding)).astype(np.float16)

# def preprocess_input(text: str) -> str:
#     return " ".join(text.lower().strip().split())

# def delete_existing_data():
#     """Delete all data from collections."""
#     try:
#         with client.start_session() as session:
#             with session.start_transaction():
#                 embeddings_collection.drop_indexes()
#                 users_collection.delete_many({}, session=session)
#                 conversations.delete_many({}, session=session)
#                 journal_collection.delete_many({}, session=session)
#                 embeddings_collection.delete_many({}, session=session)
#                 saved_greetings.delete_many({}, session=session)
#                 greetings_cache.delete_many({}, session=session)
#                 logger.info("All data deleted")
#     except Exception as e:
#         logger.error(f"Failed to delete data: {str(e)}")
#         raise

# def create_users():
#     """Create user data."""
#     users = [
#         {"user_id": "user1", "name": "Alice", "email": "alice@example.com", "created_at": datetime(2025, 5, 1)},
#         {"user_id": "user2", "name": "Bob", "email": "bob@example.com", "created_at": datetime(2025, 5, 1)},
#         {"user_id": "user3", "name": "Charlie", "email": "charlie@example.com", "created_at": datetime(2025, 5, 1)},
#         {"user_id": "user4", "name": "Diana", "email": "diana@example.com", "created_at": datetime(2025, 5, 1)},
#     ]
#     try:
#         with client.start_session() as session:
#             with session.start_transaction():
#                 users_collection.create_index("user_id", unique=True)
#                 for user in users:
#                     users_collection.update_one(
#                         {"user_id": user["user_id"]},
#                         {"$set": user},
#                         upsert=True,
#                         session=session
#                     )
#                 logger.info(f"Inserted {len(users)} users")
#     except Exception as e:
#         logger.error(f"Failed to create users: {str(e)}")
#         raise

# def create_conversations():
#     """Create conversations with single thread per user pair."""
#     conversation_data = [
#         # Alice (user1) and Bob (user2) - siblings
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_ids": ["user1", "user2"],
#             "messages": [
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user2", "target_name": "Bob", "content": "Yo, bro, struggling with math!", "timestamp": datetime(2025, 5, 1, 8, 0)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user1", "target_name": "Alice", "content": "Sis, let’s tackle it together!", "timestamp": datetime(2025, 5, 1, 8, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user2", "target_name": "Bob", "content": "Aced my math quiz!", "timestamp": datetime(2025, 5, 2, 10, 0)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user1", "target_name": "Alice", "content": "Nice job, sis!", "timestamp": datetime(2025, 5, 2, 10, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user2", "target_name": "Bob", "content": "Wanna grab pizza?", "timestamp": datetime(2025, 5, 2, 12, 0)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user1", "target_name": "Alice", "content": "Pepperoni, let’s go!", "timestamp": datetime(2025, 5, 2, 12, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user2", "target_name": "Bob", "content": "Feeling stressed about school.", "timestamp": datetime(2025, 5, 3, 8, 0)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user1", "target_name": "Alice", "content": "Chill, you got this!", "timestamp": datetime(2025, 5, 3, 8, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user2", "target_name": "Bob", "content": "Thinking of studying in Japan.", "timestamp": datetime(2025, 5, 4, 9, 0)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user1", "target_name": "Alice", "content": "That’s dope! I’m thinking Europe.", "timestamp": datetime(2025, 5, 4, 9, 5)},
#             ]
#         },
#         # Bob (user2) and Charlie (user3) - father-son
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_ids": ["user2", "user3"],
#             "messages": [
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user3", "target_name": "Charlie", "content": "Dad, advice on Alice’s studies?", "timestamp": datetime(2025, 5, 1, 10, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user2", "target_name": "Bob", "content": "Keep cheering her on, son!", "timestamp": datetime(2025, 5, 1, 10, 5)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user3", "target_name": "Charlie", "content": "You fishing this weekend?", "timestamp": datetime(2025, 5, 1, 11, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user2", "target_name": "Bob", "content": "Yup, join me?", "timestamp": datetime(2025, 5, 1, 11, 5)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user3", "target_name": "Charlie", "content": "Work’s been rough, Dad.", "timestamp": datetime(2025, 5, 2, 9, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user2", "target_name": "Bob", "content": "Let’s talk over dinner.", "timestamp": datetime(2025, 5, 2, 9, 5)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user3", "target_name": "Charlie", "content": "Still got that guitar?", "timestamp": datetime(2025, 5, 3, 10, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user2", "target_name": "Bob", "content": "Yup, let’s jam!", "timestamp": datetime(2025, 5, 3, 10, 5)},
#                 {"speaker_id": "user2", "speaker_name": "Bob", "target_id": "user3", "target_name": "Charlie", "content": "Taking Alice camping soon.", "timestamp": datetime(2025, 5, 4, 9, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user2", "target_name": "Bob", "content": "She’ll love it!", "timestamp": datetime(2025, 5, 4, 9, 5)},
#             ]
#         },
#         # Alice (user1) and Charlie (user3) - father-daughter
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_ids": ["user1", "user3"],
#             "messages": [
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user3", "target_name": "Charlie", "content": "Dad, I aced my math exam!", "timestamp": datetime(2025, 5, 1, 12, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user1", "target_name": "Alice", "content": "Proud of you, kiddo!", "timestamp": datetime(2025, 5, 1, 12, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user3", "target_name": "Charlie", "content": "Wanna go biking this weekend?", "timestamp": datetime(2025, 5, 1, 13, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user1", "target_name": "Alice", "content": "Trail by the park?", "timestamp": datetime(2025, 5, 1, 13, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user3", "target_name": "Charlie", "content": "Nervous about my science project.", "timestamp": datetime(2025, 5, 2, 8, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user1", "target_name": "Alice", "content": "You’ll crush it!", "timestamp": datetime(2025, 5, 2, 8, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user3", "target_name": "Charlie", "content": "Feeling down today.", "timestamp": datetime(2025, 5, 3, 10, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user1", "target_name": "Alice", "content": "Ice cream and a chat?", "timestamp": datetime(2025, 5, 3, 10, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user3", "target_name": "Charlie", "content": "Thinking of studying abroad.", "timestamp": datetime(2025, 5, 4, 11, 0)},
#                 {"speaker_id": "user3", "speaker_name": "Charlie", "target_id": "user1", "target_name": "Alice", "content": "Where to, kid?", "timestamp": datetime(2025, 5, 4, 11, 5)},
#             ]
#         },
#         # Alice (user1) and Diana (user4) - mother-daughter
#         {
#             "conversation_id": str(uuid.uuid4()),
#             "user_ids": ["user1", "user4"],
#             "messages": [
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user4", "target_name": "Diana", "content": "Mom, I aced my math exam!", "timestamp": datetime(2025, 5, 1, 14, 0)},
#                 {"speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Alice", "content": "That’s my girl!", "timestamp": datetime(2025, 5, 1, 14, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user4", "target_name": "Diana", "content": "Can we go shopping?", "timestamp": datetime(2025, 5, 2, 11, 0)},
#                 {"speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Alice", "content": "Mall trip, let’s do it!", "timestamp": datetime(2025, 5, 2, 11, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user4", "target_name": "Diana", "content": "School’s stressing me out.", "timestamp": datetime(2025, 5, 3, 12, 0)},
#                 {"speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Alice", "content": "Let’s talk over coffee, sweetie.", "timestamp": datetime(2025, 5, 3, 12, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user4", "target_name": "Diana", "content": "You into photography?", "timestamp": datetime(2025, 5, 4, 13, 0)},
#                 {"speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Alice", "content": "Love it! Got a new camera.", "timestamp": datetime(2025, 5, 4, 13, 5)},
#                 {"speaker_id": "user1", "speaker_name": "Alice", "target_id": "user4", "target_name": "Diana", "content": "Wanna start a blog?", "timestamp": datetime(2025, 5, 5, 10, 0)},
#                 {"speaker_id": "user4", "speaker_name": "Diana", "target_id": "user1", "target_name": "Alice", "content": "Sounds fun, I’m in!", "timestamp": datetime(2025, 5, 5, 10, 5)},
#             ]
#         },
#     ]

#     try:
#         with client.start_session() as session:
#             with session.start_transaction():
#                 conversations.create_index("conversation_id", unique=True)
#                 message_counter = 0
#                 for conv in conversation_data:
#                     conv_doc = {
#                         "conversation_id": conv["conversation_id"],
#                         "user_ids": conv["user_ids"],
#                         "messages": [
#                             {
#                                 "speaker_id": msg["speaker_id"],
#                                 "speaker_name": msg["speaker_name"],
#                                 "target_id": msg["target_id"],
#                                 "target_name": msg["target_name"],
#                                 "content": msg["content"],
#                                 "type": "user_input" if msg["speaker_id"] == conv["user_ids"][0] else "response",
#                                 "source": "human",
#                                 "timestamp": msg["timestamp"],
#                             } for msg in conv["messages"]
#                         ]
#                     }
#                     conversations.insert_one(conv_doc, session=session)
#                     for msg in conv["messages"]:
#                         message_counter += 1
#                         embedding = quantize_embedding(retriever.encode(preprocess_input(msg["content"]), normalize_embeddings=True))
#                         embeddings_collection.insert_one({
#                             "item_id": f"{conv['conversation_id']}_{msg['speaker_id']}_{msg['target_id']}_{message_counter}",
#                             "item_type": "conversation",
#                             "user_id": conv["user_ids"],
#                             "speaker_id": msg["speaker_id"],
#                             "speaker_name": msg["speaker_name"],
#                             "target_id": msg["target_id"],
#                             "target_name": msg["target_name"],
#                             "timestamp": msg["timestamp"],
#                             "embedding": embedding.tolist(),
#                         }, session=session)
#                 logger.info(f"Inserted {conversations.count_documents({}, session=session)} conversations")
#     except Exception as e:
#         logger.error(f"Failed to create conversations: {str(e)}")
#         raise

# def create_journals():
#     """Create journal entries for each user."""
#     journal_data = {
#         "user1": [
#             {"entry_id": "journal1", "content": "Pumped about my math exam!", "timestamp": datetime(2025, 5, 1, 15, 0)},
#             {"entry_id": "journal2", "content": "School’s stressful, but Bob’s supportive.", "timestamp": datetime(2025, 5, 2, 20, 0)},
#             {"entry_id": "journal3", "content": "Biking with Dad was fun!", "timestamp": datetime(2025, 5, 3, 18, 0)},
#             {"entry_id": "journal4", "content": "Dreaming of studying in Japan.", "timestamp": datetime(2025, 5, 4, 21, 0)},
#             {"entry_id": "journal5", "content": "Planning a blog with Mom!", "timestamp": datetime(2025, 5, 5, 22, 0)},
#         ],
#         "user2": [
#             {"entry_id": "journal6", "content": "Helped Alice with math today.", "timestamp": datetime(2025, 5, 1, 16, 0)},
#             {"entry_id": "journal7", "content": "Work’s tough, Dad’s advice helps.", "timestamp": datetime(2025, 5, 2, 21, 0)},
#             {"entry_id": "journal8", "content": "Planning a camping trip.", "timestamp": datetime(2025, 5, 3, 19, 0)},
#             {"entry_id": "journal9", "content": "Jammed with Dad, good vibes.", "timestamp": datetime(2025, 5, 4, 22, 0)},
#             {"entry_id": "journal10", "content": "Pizza with Alice was awesome.", "timestamp": datetime(2025, 5, 5, 23, 0)},
#         ],
#         "user3": [
#             {"entry_id": "journal11", "content": "Proud of Alice’s math exam!", "timestamp": datetime(2025, 5, 1, 17, 0)},
#             {"entry_id": "journal12", "content": "Bob’s stressed, planning dinner.", "timestamp": datetime(2025, 5, 2, 22, 0)},
#             {"entry_id": "journal13", "content": "Biking with Alice was great.", "timestamp": datetime(2025, 5, 3, 20, 0)},
#             {"entry_id": "journal14", "content": "Fishing with Bob, good times.", "timestamp": datetime(2025, 5, 4, 18, 0)},
#             {"entry_id": "journal15", "content": "Alice’s abroad plans, big dreams!", "timestamp": datetime(2025, 5, 5, 21, 0)},
#         ],
#         "user4": [
#             {"entry_id": "journal16", "content": "Alice aced her exam, so proud!", "timestamp": datetime(2025, 5, 1, 18, 0)},
#             {"entry_id": "journal17", "content": "Shopping with Alice was fun.", "timestamp": datetime(2025, 5, 2, 23, 0)},
#             {"entry_id": "journal18", "content": "Coffee with Alice, she’s stressed.", "timestamp": datetime(2025, 5, 3, 21, 0)},
#             {"entry_id": "journal19", "content": "Loving my new camera!", "timestamp": datetime(2025, 5, 4, 20, 0)},
#             {"entry_id": "journal20", "content": "Blog idea with Alice is exciting.", "timestamp": datetime(2025, 5, 5, 19, 0)},
#         ],
#     }

#     try:
#         with client.start_session() as session:
#             with session.start_transaction():
#                 journal_collection.create_index("entry_id", unique=True)
#                 for user_id, entries in journal_data.items():
#                     user = users_collection.find_one({"user_id": user_id}, session=session)
#                     for entry in entries:
#                         journal_doc = {
#                             "entry_id": entry["entry_id"],
#                             "user_id": [user_id],
#                             "speaker_id": user_id,
#                             "speaker_name": user["name"],
#                             "content": entry["content"],
#                             "timestamp": entry["timestamp"],
#                         }
#                         journal_collection.insert_one(journal_doc, session=session)
#                         embedding = quantize_embedding(retriever.encode(preprocess_input(entry["content"]), normalize_embeddings=True))
#                         embeddings_collection.insert_one({
#                             "item_id": entry["entry_id"],
#                             "item_type": "journal",
#                             "user_id": [user_id],
#                             "speaker_id": user_id,
#                             "speaker_name": user["name"],
#                             "timestamp": entry["timestamp"],
#                             "embedding": embedding.tolist(),
#                         }, session=session)
#                 logger.info(f"Inserted {journal_collection.count_documents({}, session=session)} journal entries")
#     except Exception as e:
#         logger.error(f"Failed to create journals: {str(e)}")
#         raise

# def create_greetings():
#     """Create greeting data."""
#     greetings = [
#         {"greeting_id": str(uuid.uuid4()), "target_id": "user2", "user_id": "user1", "bot_role": "brother", "greeting": "Yo, bro!", "tone": "playful", "timestamp": datetime(2025, 5, 1)},
#         {"greeting_id": str(uuid.uuid4()), "target_id": "user3", "user_id": "user2", "bot_role": "father", "greeting": "Hey, Dad!", "tone": "warm", "timestamp": datetime(2025, 5, 1)},
#         {"greeting_id": str(uuid.uuid4()), "target_id": "user3", "user_id": "user1", "bot_role": "father", "greeting": "Hey, Dad!", "tone": "youthful", "timestamp": datetime(2025, 5, 1)},
#         {"greeting_id": str(uuid.uuid4()), "target_id": "user4", "user_id": "user1", "bot_role": "mother", "greeting": "Hey, Mom!", "tone": "warm", "timestamp": datetime(2025, 5, 1)},
#     ]
#     try:
#         with client.start_session() as session:
#             with session.start_transaction():
#                 saved_greetings.create_index("greeting_id", unique=True)
#                 for greeting in greetings:
#                     saved_greetings.insert_one(greeting, session=session)
#                 logger.info(f"Inserted {saved_greetings.count_documents({}, session=session)} greetings")
#     except Exception as e:
#         logger.error(f"Failed to create greetings: {str(e)}")
#         raise

# def main():
#     """Run all data setup functions."""
#     try:
#         delete_existing_data()
#         create_users()
#         create_conversations()
#         create_journals()
#         create_greetings()
#         logger.info("Data setup completed")
#     except Exception as e:
#         logger.error(f"Data setup failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()




# from pymongo import MongoClient
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# client = MongoClient(
#     "mongodb+srv://tamannabdcalling4:kfQTC9evfsxzIaFa@cluster0.fnawduk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
#     tls=True,
#     tlsAllowInvalidCertificates=True
# )
# db = client["user_bots"]

# def clear_all_collections():
#     logger.info("Clearing all collections in user_bots database")
#     collections = ["users", "conversations", "embeddings", "personalities", "saved_greetings"]
#     for collection_name in collections:
#         try:
#             db[collection_name].delete_many({})
#             logger.info(f"Cleared collection: {collection_name}")
#         except Exception as e:
#             logger.error(f"Failed to clear {collection_name}: {str(e)}")

# if __name__ == "__main__":
#     clear_all_collections()
#     logger.info("All collections cleared successfully")




# import os
# from pymongo import MongoClient
# from datetime import datetime
# import logging
# from dotenv import load_dotenv

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()
# MONGODB_URI = os.getenv("MONGODB_URI")
# if not MONGODB_URI:
#     logger.error("MONGODB_URI not found in .env file")
#     raise ValueError("MONGODB_URI not found")

# # MongoDB setup
# logger.info("Connecting to MongoDB Atlas")
# client = MongoClient(MONGODB_URI, tls=True, tlsAllowInvalidCertificates=True)
# db = client["user_bots"]
# users_collection = db["users"]
# conversations = db["conversations"]
# journal_collection = db["journal_entries"]
# embeddings_collection = db["embeddings"]
# saved_greetings = db["saved_greetings"]

# # Additional demo conversations
# additional_conversations = [
#     {
#         "conversation_id": "conv14",
#         "user_id": ["user1", "user2"],
#         "speaker_id": "user1",
#         "speaker_name": "Alice",
#         "target_id": "user2",
#         "target_name": "Bob",
#         "content": "Bob, let's plan a movie night this weekend!",
#         "type": "user_input",
#         "source": "human",
#         "timestamp": datetime(2025, 5, 13, 14, 0, 0)
#     },
#     {
#         "conversation_id": "conv15",
#         "user_id": ["user1", "user2"],
#         "speaker_id": "user2",
#         "speaker_name": "Bob",
#         "target_id": "user1",
#         "target_name": "Alice",
#         "content": "Sweet, I'm in! Action or comedy?",
#         "type": "user_input",
#         "source": "human",
#         "timestamp": datetime(2025, 5, 13, 14, 5, 0)
#     },
#     {
#         "conversation_id": "conv16",
#         "user_id": ["user1", "user2"],
#         "speaker_id": "user1",
#         "speaker_name": "Alice",
#         "target_id": "user2",
#         "target_name": "Bob",
#         "content": "Let's go with comedy. Got any favorites?",
#         "type": "user_input",
#         "source": "human",
#         "timestamp": datetime(2025, 5, 13, 14, 10, 0)
#     },
#     {
#         "conversation_id": "conv17",
#         "user_id": ["user3", "user4"],
#         "speaker_id": "user3",
#         "speaker_name": "Charlie",
#         "target_id": "user4",
#         "target_name": "Diana",
#         "content": "Diana, the kids are planning a movie night. Should we join?",
#         "type": "user_input",
#         "source": "human",
#         "timestamp": datetime(2025, 5, 13, 15, 0, 0)
#     },
# ]

# # Additional demo journal entries
# additional_journals = {
#     "user1": [
#         {"entry_id": "journal12", "content": "I love comedy and  thriller movie!", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
#         {"entry_id": "journal13", "content": "Feeling a bit nervous about my presentation tomorrow.", "timestamp": datetime(2025, 5, 14, 9, 0, 0)},
#     ],
#     "user2": [
#         {"entry_id": "journal14", "content": "Alice’s movie night idea is awesome. Gotta pick a good comedy.", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
#     ],
#     "user3": [
#         {"entry_id": "journal15", "content": "The kids are growing up so fast. Movie night sounds like a great family plan.", "timestamp": datetime(2025, 5, 13, 9, 0, 0)},
#     ],
# }

# def add_demo_data():
#     logger.info("Starting add_demo_data")
    
#     # Ensure demo users exist
#     demo_users = [
#         {"user_id": "user1", "name": "Alice"},
#         {"user_id": "user2", "name": "Bob"},
#         {"user_id": "user3", "name": "Charlie"},
#         {"user_id": "user4", "name": "Diana"}
#     ]
#     for user in demo_users:
#         if not users_collection.find_one({"user_id": user["user_id"]}):
#             users_collection.insert_one(user)
#             logger.debug(f"Inserted user: {user['user_id']}")

#     # Insert additional conversations
#     for conv in additional_conversations:
#         if conversations.find_one({"conversation_id": conv["conversation_id"]}):
#             logger.debug(f"Skipping existing conversation: {conv['conversation_id']}")
#             continue
#         conversations.insert_one(conv)
#         logger.debug(f"Inserted conversation: {conv['conversation_id']}")

#     # Insert additional journal entries
#     for user_id, entries in additional_journals.items():
#         user = users_collection.find_one({"user_id": user_id})
#         if not user:
#             logger.error(f"User {user_id} not found")
#             continue
#         for entry in entries:
#             if journal_collection.find_one({"entry_id": entry["entry_id"]}):
#                 logger.debug(f"Skipping existing journal: {entry['entry_id']}")
#                 continue
#             journal_doc = {
#                 "entry_id": entry["entry_id"],
#                 "user_id": [user_id],
#                 "speaker_id": user_id,
#                 "speaker_name": user["name"],
#                 "content": entry["content"],
#                 "timestamp": entry["timestamp"]
#             }
#             journal_collection.insert_one(journal_doc)
#             logger.debug(f"Inserted journal: {entry['entry_id']} for user {user_id}")

#     logger.info(f"Conversations count: {conversations.count_documents({})}")
#     logger.info(f"Journal entries count: {journal_collection.count_documents({})}")

# if __name__ == "__main__":
#     add_demo_data()





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