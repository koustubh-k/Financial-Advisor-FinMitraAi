
# ====================
# 3. database.py
# ====================
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from config import Config
import os

class UserHistoryVectorDB:
    """Manages user chat history using a persistent vector store."""
    def __init__(self, collection_name="user_chat_history", embedding_model=None):
        if embedding_model is None:
            if Config.HF_TOKEN and Config.HF_TOKEN != "":
                os.environ["HF_TOKEN"] = Config.HF_TOKEN
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            self.embedding_model = embedding_model
        
        self.collection_name = collection_name
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=Config.VECTOR_STORE_PATH
        )

    def add_message(self, user_id: str, message: str):
        """Adds a new message to the user's history."""
        try:
            self.vector_store.add_texts(
                texts=[message],
                metadatas=[{"user_id": user_id}]
            )
            print(f"Added message to ChromaDB for user {user_id}.")
        except Exception as e:
            print(f"Error adding message to ChromaDB: {e}")

    def retrieve_history(self, user_id: str, query: str, k: int = 5) -> List[str]:
        """Retrieves relevant history for a given user and query."""
        try:
            docs = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter={"user_id": user_id}
            )
            return [doc[0].page_content for doc in docs]
        except Exception as e:
            print(f"Error retrieving history from ChromaDB: {e}")
            return []

