# ==============================================================================
# MCP Financial Advisor Bot
# A modular Python agent using LangChain and LangGraph for a hackathon project.
# This version is corrected for local testing with a FastAPI server but without
# Twilio-specific WhatsApp integration.
#
# To run this code, you will need to install the necessary libraries:
# pip install fastapi uvicorn langchain langchain-groq langchain-chromadb python-dotenv requests reportlab langchain_huggingface
#
# Remember to configure your API keys in a .env file.
# ==============================================================================

# ====================
# 1. config.py
# ====================
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Config:
    """Configuration class for API keys and settings."""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gsk_1234567890")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "123")
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")  # Alternative stock API
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # For news data
    HF_TOKEN = os.getenv("HF_TOKEN", "cxQn@1WFAzKFHMGhxyKiGwaFrbmoOXuxcDk")
    VECTOR_STORE_PATH = "chromadb_user_history"
    
    # API URLs
    ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
    FINNHUB_URL = "https://finnhub.io/api/v1"
    NEWS_API_URL = "https://newsapi.org/v2"
    NSE_API_URL = "https://www.nseindia.com/api"
