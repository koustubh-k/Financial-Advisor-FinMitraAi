# Fin_Adv: AI-Powered Financial Advisor Bot

An advanced AI financial advisor specializing in the Indian stock market with real-time data capabilities, built using LangChain, LangGraph, and modern web technologies.

## 🚀 Features

### Core Capabilities

- **Real-time Market Data**: Live Nifty 50, stock prices, and gold rates via Yahoo Finance
- **Intelligent Web Search**: Financial news and market analysis using DuckDuckGo
- **Portfolio Analysis**: Comprehensive portfolio evaluation with current market prices
- **Market Alerts**: Custom price alerts for Nifty 50 and stocks
- **Real Estate Insights**: Property market analysis and investment guidance
- **PDF Report Generation**: Automated market analysis reports

### AI & ML Features

- **Conversational AI**: Powered by Groq's Llama 3.1 8B model
- **Contextual Memory**: Persistent chat history using ChromaDB vector storage
- **Multi-agent Architecture**: LangGraph workflow for complex financial queries
- **Fallback Systems**: Robust error handling with multiple data source fallbacks

### User Interfaces

- **FastAPI Backend**: RESTful API for programmatic access
- **Streamlit Web UI**: Interactive web interface for users
- **WhatsApp Integration**: Direct messaging support via webhooks

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **AI/ML**: LangChain, LangGraph, Groq API, HuggingFace Embeddings
- **Database**: ChromaDB for vector storage
- **Data Sources**: Yahoo Finance, DuckDuckGo Search, NSE API
- **Frontend**: Streamlit for web UI
- **Deployment**: Docker, cloud platforms (AWS/Heroku)

## 📋 Prerequisites

- Python 3.10 or higher
- Valid API keys for:
  - Groq API (for LLM)
  - HuggingFace (for embeddings)
- Internet connection for real-time data

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd fin_adv
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv fin_venv
source fin_venv/bin/activate  # On Windows: fin_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy template and edit
cp env_template.txt .env

# Edit .env with your API keys
# Required: GROQ_API_KEY, HF_TOKEN
```

### 4. Run the Application

#### Option A: FastAPI Server Only

```bash
python -m app.main
# Server starts at http://localhost:8000
```

#### Option B: Streamlit Web UI

```bash
streamlit run app/streamlit_app.py
# Opens web interface in browser
```

#### Option C: Both Services (Recommended)

```bash
# Terminal 1: Start FastAPI backend
python -m app.main

# Terminal 2: Start Streamlit frontend
streamlit run app/streamlit_app.py
```

## 🔧 Configuration

### API Keys Setup

Create a `.env` file in the root directory:

```env
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here

# Optional API Keys (enhance functionality)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
```

### Model Configuration

- Default: Groq Llama 3.1 8B (fast, cost-effective)
- Alternative: Local Ollama models (privacy-focused)

## 📖 Usage

### Web Interface (Streamlit)

1. Open the Streamlit app in your browser
2. Enter your user ID for personalized experience
3. Ask financial questions in natural language
4. View real-time market data and analysis

### API Usage (FastAPI)

```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Send message
data = {
    "user_id": "user123",
    "message_body": "What's the current Nifty level?"
}
response = requests.post("http://localhost:8000/webhook", data=data)
print(response.json())
```

### Example Queries

- "Show me the latest Nifty data with current news"
- "What's the current price of Reliance stock?"
- "Analyze my portfolio: [{'ticker': 'RELIANCE', 'quantity': 100}]"
- "Set an alert when Nifty hits 22,500"
- "What's the real estate market like in Mumbai?"

## 🏗️ Project Structure

```
fin_adv/
├── config.py                 # Configuration management
├── models/                   # AI model management
│   ├── __init__.py
│   └── llm.py
├── database/                 # Data persistence
│   ├── __init__.py
│   └── user_history.py
├── tools/                    # Financial tools
│   ├── __init__.py
│   └── financial_tools.py
├── app/                      # Application layer
│   ├── __init__.py
│   ├── main.py              # FastAPI backend
│   └── streamlit_app.py     # Web UI
├── chromadb_user_history/    # Vector database
├── fin_venv/                # Virtual environment
├── requirements.txt          # Dependencies
├── README.md                # This file
├── setup.md                 # Detailed code explanations
└── project_structure.txt    # Architecture overview
```

## 🚀 Deployment

### Local Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Manual deployment
python -m app.main &
streamlit run app/streamlit_app.py &
```

### Cloud Deployment

#### Heroku

```bash
# Install Heroku CLI
heroku create your-app-name
heroku config:set GROQ_API_KEY=your_key
git push heroku main
```

#### AWS/DigitalOcean

- Use Docker containers
- Configure environment variables
- Set up load balancer for both services

### Docker Deployment

```bash
# Build and run
docker build -t fin-adv .
docker run -p 8000:8000 -p 8501:8501 fin-adv
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It does not constitute financial advice. Always consult with qualified financial advisors before making investment decisions. The developers are not responsible for any financial losses incurred through the use of this application.

## 📞 Support

For questions, issues, or contributions:

- Open an issue on GitHub
- Check the setup.md for detailed technical documentation
- Review the TODO.md for planned features and improvements
