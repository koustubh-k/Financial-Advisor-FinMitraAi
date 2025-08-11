import asyncio
from typing import TypedDict, Annotated, Sequence, Optional
import operator
import os
import datetime
from dotenv import load_dotenv

# --- Langchain and LangGraph imports ---
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

# --- MCP imports ---
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import Field, BaseModel

# --- FastMCP handles the web server, so FastAPI is not needed directly ---
import uvicorn

# --- Enhanced tools and utility imports ---
from enhanced_tools import (
    get_nifty_data, perform_web_search, generate_pdf_report,
    analyze_portfolio, set_nifty_alert, get_stock_price,
    get_real_estate_info, get_gold_price
)
from models import get_llm_model
from database import UserHistoryVectorDB

# --- Load environment variables ---
load_dotenv()
TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")
assert TOKEN, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
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

# --- Initialize MCP (This object serves as the main application) ---
mcp = FastMCP(
    "FinMitra AI Financial Advisor",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Required MCP tool ---
@mcp.tool
def validate() -> str:
    """A required tool that returns the server's phone number for validation."""
    return MY_NUMBER

# --- Agent and LangGraph Setup ---
class AgentState(TypedDict):
    input: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    agent_outcome: AgentAction | AgentFinish | None
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    user_id: str
    history_docs: str

tools = [
    get_nifty_data, perform_web_search, generate_pdf_report,
    analyze_portfolio, set_nifty_alert, get_stock_price,
    get_real_estate_info, get_gold_price
]

model_choice = os.getenv("LLM_MODEL_CHOICE", "groq")
llm = get_llm_model(model_choice=model_choice)

try:
    db = UserHistoryVectorDB()
except OSError as e:
    if "os error 1455" in str(e).lower():
        print("Fatal Error: Insufficient virtual memory.")
        exit(1)
    else:
        raise

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an advanced AI financial advisor bot specializing in the Indian stock market.
    - Access live market data
    - Provide actionable insights
    - Include timestamps and sources
    - Maintain a professional yet approachable tone
    User's recent chat history: {history_docs}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent(state: AgentState):
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
        return {"agent_outcome": f"⚠️ Error: {str(e)}"}

workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_edge("agent", END)
workflow.set_entry_point("agent")
app_graph = workflow.compile()


# --- NEW: MCP Tool to run the entire agent logic ---
class FinancialAdvisorDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

FinancialAdvisorQueryDescription = FinancialAdvisorDescription(
    description="A specialized tool to answer user queries about the Indian stock market, personal finance, and market data.",
    use_when="The user asks for real-time stock prices, market news, portfolio analysis, or general financial advice.",
    side_effects="Returns a comprehensive analysis based on real-time market data.",
)

@mcp.tool(description=FinancialAdvisorQueryDescription.model_dump_json())
async def financial_advisor_query(
    user_query: Annotated[str, Field(description="The user's free-form query about finance or markets.")],
    puch_user_id: Annotated[str, Field(description="The unique identifier for the user provided by Puch.")]
) -> str:
    """
    Analyzes and responds to a user's financial query by running a specialized AI agent.
    """
    state = {
        "input": user_query,
        "chat_history": [],
        "user_id": puch_user_id,
        "intermediate_steps": [],
        "history_docs": ""
    }
    try:
        response = app_graph.invoke(state)
        return response.get('agent_outcome', "No response from agent.")
    except Exception as e:
        return f"⚠️ Technical issue: {str(e)}"

# --- Main Application Loop (FastMCP handles app and routing) ---
async def main():
    print("🚀 Starting FinMitra AI Financial Advisor MCP Server on http://0.0.0.0:8086")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8086)

if __name__ == "__main__":
    asyncio.run(main())
