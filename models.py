
# ====================
# 4. models.py
# ====================
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from config import Config

def get_llm_model(model_choice="groq") -> BaseChatModel:
    """
    Returns an instance of the configured LLM.
    You can choose between a Groq model or a local Ollama model.
    """
    print(f"Initializing LLM model: {model_choice}...")
    try:
        if model_choice == "groq":
            return ChatGroq(
                temperature=0,
                groq_api_key=Config.GROQ_API_KEY,
                model_name="llama3-8b-8192"
            )
        elif model_choice == "ollama":
            return ChatOllama(model="llama3")
        else:
            raise ValueError("Invalid model choice. Use 'groq' or 'ollama'.")
    except Exception as e:
        print(f"Error initializing LLM model: {e}")
        raise e
