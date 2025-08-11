# ====================
# 6. Enhanced main application with puch_user_id authentication
# ====================
import asyncio
from typing import TypedDict, Annotated, Sequence, Optional, Literal
import operator
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from fastapi import FastAPI, Request, Form
import uvicorn
from contextlib import asynccontextmanager
import datetime
import os
import json

# Import necessary modules for authentication
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import Field, BaseModel

from dotenv import load_dotenv

# Import enhanced tools
from enhanced_tools import (
    get_nifty_data, perform_web_search, generate_pdf_report,
    analyze_portfolio, set_nifty_alert, get_stock_price,
    get_real_estate_info, get_gold_price
)
from models import get_llm_model
from database import UserHistoryVectorDB
from config import Config

# Load environment variables
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER, "Please set MY_NUMBER in your .env file"

# --- Auth Provider with puch_user_id ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token, client_id="finmitra-client", scopes=["*"], expires_at=None
            )
        return None

mcp = FastMCP(
    "FinMitra AI Financial Advisor",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- MCP Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

class AgentState(TypedDict):
    input: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    agent_outcome: AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    user_id: str
    history_docs: str

# Enhanced tools list with real-time capabilities
tools = [
    get_nifty_data, perform_web_search, generate_pdf_report,
    analyze_portfolio, set_nifty_alert, get_stock_price,
    get_real_estate_info, get_gold_price
]

# Use a default model if not configured
model_choice = os.getenv("LLM_MODEL_CHOICE", "groq")
llm = get_llm_model(model_choice=model_choice)

try:
    db = UserHistoryVectorDB()
except OSError as e:
    if "os error 1455" in str(e).lower():
        print("Fatal Error: Insufficient virtual memory. The program requires more memory to load the embedding model.")
        print("Please consider one of the following solutions:")
        print("1. Increase your system's paging file size (virtual memory).")
        print("2. Use a smaller embedding model, e.g., a lightweight one instead of 'all-MiniLLM-L6-v2'.")
        exit(1)
    else:
        raise

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced AI financial advisor bot specializing in the Indian stock market with real-time data capabilities.

    Your key features:
    - Access to live market data via Yahoo Finance and web search
    - Real-time Nifty 50, stock prices, and gold rates
    - Current financial news through DuckDuckGo search
    - Comprehensive portfolio analysis with live prices
    - Market sentiment analysis based on current data

    Guidelines:
    - Always use real-time tools for current market data
    - Provide actionable insights based on live information
    - Include timestamps and data sources in your responses
    - Combine multiple data sources for comprehensive analysis
    - Be proactive in suggesting relevant follow-up actions
    - Maintain a professional yet approachable tone

    User's recent chat history: {history_docs}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(state: AgentState):
    """Enhanced agent runner with better error handling."""
    try:
        history_docs = db.retrieve_history(state['user_id'], state['input'])
        history_string = "\n".join(history_docs)

        result = agent_executor.invoke({
            "input": state['input'],
            "chat_history": state['chat_history'],
            "history_docs": history_string,
            "agent_scratchpad": state['intermediate_steps']
        })

        db.add_message(state['user_id'], f"Human: {state['input']}")
        db.add_message(state['user_id'], f"AI: {result['output']}")

        return {"agent_outcome": result['output']}

    except Exception as e:
        print(f"Agent execution error: {e}")
        error_msg = str(e)

        if "over capacity" in error_msg.lower():
            return {"agent_outcome": "🚫 The AI service is currently at capacity. Please try again in a few moments!"}
        elif "timeout" in error_msg.lower():
            return {"agent_outcome": "⏱️ Request timed out. Please try again with a simpler query."}
        else:
            return {"agent_outcome": f"⚠️ I encountered an issue: {error_msg}. Please try rephrasing your question."}

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_edge("agent", END)
workflow.set_entry_point("agent")
app_graph = workflow.compile()

def handle_message(user_id: str, message: str):
    """Enhanced message handler with real-time processing."""
    print(f"\n🔄 Processing message from User {user_id}")
    print(f"Query: {message}")

    state = {
        "input": message,
        "chat_history": [],
        "user_id": user_id,
        "intermediate_steps": [],
        "history_docs": ""
    }

    try:
        response = app_graph.invoke(state)
        final_output = response['agent_outcome']
    except Exception as e:
        print(f"Message handling error: {e}")
        final_output = "⚠️ I'm experiencing technical difficulties. Please try again in a moment."

    print(f"Response: {final_output}")
    print("✅ Processing complete\n")
    return final_output

app = FastAPI(title="Enhanced Financial Advisor Bot", version="2.0")

@app.get("/")
async def health_check():
    """Enhanced health check with system status."""
    return {
        "status": "healthy",
        "version": "2.0",
        "features": [
            "Real-time market data",
            "DuckDuckGo web search",
            "Live stock prices",
            "Portfolio analysis",
            "Market news integration"
        ],
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.post("/webhook")
async def enhanced_webhook(user_id: str = Form(...), message_body: str = Form(...)):
    """
    Enhanced webhook endpoint with real-time processing capabilities.
    """
    print(f"📨 Webhook received from {user_id}: {message_body}")

    try:
        response_message = handle_message(user_id, message_body)

        return {
            "response": response_message,
            "status": "success",
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "features_used": ["real_time_data", "web_search"]
        }

    except Exception as e:
        print(f"Webhook error: {e}")
        return {
            "response": "⚠️ Service temporarily unavailable. Please try again.",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

@app.get("/market-status")
async def get_market_status():
    """
    API endpoint to check current market status.
    """
    try:
        from enhanced_tools import RealTimeDataProvider
        nse_data = RealTimeDataProvider.get_nse_data()
        return {
            "status": "success",
            "market_data": nse_data,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }

if __name__ == "__main__":
    print("🚀 Starting FinMitra AI Financial Advisor Bot...")
    print("📊 Features: Real-time data, Web search, Live prices")

    # Start the FastAPI server
    uvicorn.run("mcp_main:app", host="0.0.0.0", port=8000, reload=True)
