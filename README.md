# Euphoria – Advanced Chainlit AI Assistant

A powerful conversational AI assistant built with **Chainlit** and **LangChain**. Features intelligent query routing that automatically chooses the best response method:
- **RAG** from uploaded documents (FAISS vector store)
- **Real-time web search** (Tavily)
- **Math calculations**
- **Web scraping**
- **Direct LLM responses** (GPT-4o-mini)

### Key Features
- Upload documents for smart knowledge base Q&A
- Streaming responses
- Tools: web search, calculator, scrape & save pages
- Session stats, chat history view, and export
- Clean interactive UI with action buttons

### Tech Stack
- Chainlit • LangChain • OpenAI (GPT-4o-mini)
- FAISS • Tavily Search

### Quick Start
```bash
git clone https://github.com/pinankpanchal/Euphoria-my-chainlit-bot.git
cd Euphoria-my-chainlit-bot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add .env with OPENAI_API_KEY and TAVILY_API_KEY, then:

chainlit run app.py -w
