import chainlit as cl
import os
import httpx  # ✅ Changed from requests to httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Any, List, AsyncGenerator
from bs4 import BeautifulSoup
import re
from pathlib import Path
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


load_dotenv()

BACKEND_URL = "http://localhost:8000"

# ✅ Create async HTTP client
http_client = httpx.AsyncClient(timeout=30.0)

# =========================================================
# QUERY ROUTER WITH LLM-BASED CLASSIFICATION
# =========================================================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal[
        "rag_vectorstore",
        "web_search", 
        "calculator",
        "web_scraper",
        "direct_llm"
    ] = Field(
        ...,
        description="Given a user question choose which datasource would be most relevant for answering their question",
    )
    
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this datasource was chosen"
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1 for this routing decision"
    )


class AdvancedQueryRouter:
    """
    Intelligent query router that analyzes user queries and routes them
    to the most appropriate data source or tool.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create structured output LLM
        self.structured_llm = self.llm.with_structured_output(RouteQuery)
        
        # Define routing prompt
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at routing user questions to the appropriate data source.

Available data sources:
- rag_vectorstore: Use for questions about uploaded documents, internal knowledge base, or previously stored information
- web_search: Use for current events, recent news, real-time information, or topics not in the knowledge base
- calculator: Use for mathematical calculations, arithmetic operations, or numerical computations
- web_scraper: Use when user explicitly wants to extract content from a specific URL
- direct_llm: Use for general knowledge questions, creative tasks, or when no external data is needed

Consider these factors when routing:
1. Does the query reference uploaded documents or internal knowledge?
2. Does it require current/real-time information?
3. Is it a calculation or math problem?
4. Does it mention a specific URL to scrape?
5. Can it be answered with general knowledge?

Provide your reasoning and confidence score."""),
            ("human", "{question}")
        ])
        
        self.router_chain = self.routing_prompt | self.structured_llm
    
    async def route(self, query: str, has_documents: bool = False) -> RouteQuery:
        """
        Route the query to the most appropriate datasource.
        
        Args:
            query: User's query string
            has_documents: Whether RAG vector store has documents
        
        Returns:
            RouteQuery with datasource, reasoning, and confidence
        """
        # Rule-based routing for explicit patterns
        query_lower = query.lower()
        
        # Check for URL scraping
        if any(x in query_lower for x in ["scrape", "http://", "https://", "www."]):
            return RouteQuery(
                datasource="web_scraper",
                reasoning="Query contains URL or scraping keywords",
                confidence=0.95
            )
        
        # Check for calculations
        if any(x in query_lower for x in ["calculate", "compute", "what is", "equals"]) and any(c.isdigit() for c in query):
            return RouteQuery(
                datasource="calculator",
                reasoning="Query is a mathematical calculation",
                confidence=0.95
            )
        
        # Check for current/recent information needs
        current_keywords = ["today", "latest", "current", "recent", "news", "now", "2024", "2025", "yesterday", "this week"]
        if any(keyword in query_lower for keyword in current_keywords):
            return RouteQuery(
                datasource="web_search",
                reasoning="Query requires current/real-time information",
                confidence=0.90
            )
        
        # If documents exist, prefer RAG unless explicitly asking for web search
        if has_documents:
            # Check if explicitly asking for web search
            web_keywords = ["search the web", "google", "search online", "look up online"]
            if any(keyword in query_lower for keyword in web_keywords):
                return RouteQuery(
                    datasource="web_search",
                    reasoning="Explicit request for web search",
                    confidence=0.90
                )
            
            # Default to RAG when documents exist
            # Let LLM decide but bias toward RAG
            modified_prompt = f"""{query}

Note: Documents are currently loaded in the RAG system. Unless the query explicitly requires current/real-time information or web search, prefer using rag_vectorstore."""
            
            result = await self.router_chain.ainvoke({"question": modified_prompt})
            
            # If LLM chose direct_llm but we have documents, switch to RAG
            if result.datasource == "direct_llm":
                return RouteQuery(
                    datasource="rag_vectorstore",
                    reasoning="Documents available - checking RAG first before using general knowledge",
                    confidence=0.75
                )
            
            return result
        else:
            # No documents - exclude RAG option
            modified_prompt = self.routing_prompt.format_messages(
                question=f"{query}\n\nNote: No documents are currently loaded in the RAG system."
            )
            result = await self.structured_llm.ainvoke(modified_prompt)
            return result

# =========================================================
# 1. STREAMING RAG CONFIGURATION
# =========================================================

class StreamingRAG:
    def __init__(self):
        self.vector_store = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=800,
            streaming=True
        )
    
    def add_documents(self, documents: List[str], metadata: List[dict] = None):
        """Add documents to vector store."""
        doc_objects = []
        
        for i, doc_content in enumerate(documents):
            # Split document into chunks
            chunks = self.splitter.split_text(doc_content)
            
            for chunk_idx, chunk in enumerate(chunks):
                meta = metadata[i] if metadata else {}
                meta.update({
                    "doc_index": i,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                })
                
                doc_objects.append(
                    Document(page_content=chunk, metadata=meta)
                )
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(doc_objects, self.embeddings)
        else:
            self.vector_store.add_documents(doc_objects)
    
    async def retrieve_context(self, query: str, k: int = 5) -> tuple:
        """Retrieve relevant chunks with streaming chunks."""
        if self.vector_store is None:
            return [], ""
        
        # Retrieve top chunks
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        chunks = retriever.invoke(query)
        
        # Prepare context
        context_str = "\n\n---\n\n".join([
            f"[Source: {chunk.metadata.get('source', 'Document')} - "
            f"Chunk {chunk.metadata.get('chunk_index', 0)} of "
            f"{chunk.metadata.get('total_chunks', 1)}]\n\n{chunk.page_content}"
            for chunk in chunks
        ])
        
        return chunks, context_str
    
    async def stream_rag_response(
        self, 
        query: str, 
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """Stream RAG response with retrieved context."""
        
        # Retrieve context
        chunks, context = await self.retrieve_context(query)
        
        if not chunks:
            yield "No relevant documents found for your query."
            return
        
        # Build prompt with context
        if system_prompt is None:
            system_prompt = """You are a helpful assistant with access to a knowledge base.
Use the retrieved context to answer the user's question accurately and concisely.
If the context doesn't contain relevant information, say so clearly.
Always cite which document chunk you're referencing."""
        
        prompt = f"""{system_prompt}

---CONTEXT FROM KNOWLEDGE BASE---
{context}
---END CONTEXT---

User Question: {query}"""
        
        # Stream the response
        async for chunk in self.llm.astream(prompt):
            if chunk.content:
                yield chunk.content

# =========================================================
# 2. DEFINE TOOLS
# =========================================================

tavily_search = TavilySearch(max_results=5)

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily."""
    try:
        results = tavily_search.invoke(query)
        if isinstance(results, str):
            return results[:800]
        
        summary = []
        if isinstance(results, list):
            for r in results[:3]:
                if isinstance(r, dict):
                    title = r.get("title", "")
                    content = r.get("content", "")
                    url = r.get("url", "")
                    if title and content:
                        summary.append(f"{title}\n{content}\n{url}")
        
        return "\n\n".join(summary)[:800] if summary else str(results)[:800]
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def calculator(expr: str) -> str:
    """Basic arithmetic calculator."""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expr):
        return "Error: Only arithmetic symbols allowed"
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def scrape_webpage(url: str) -> str:
    """Scrape content from a webpage and extract text."""
    import requests  # Keep requests here for synchronous tool
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script in soup(['script', 'style', 'meta', 'noscript']):
            script.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    except requests.exceptions.Timeout:
        return "Error: Request timed out. URL took too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to the URL. Check if the website is accessible."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} - {e.response.reason}"
    except Exception as e:
        return f"Scraping error: {str(e)}"

@tool
def scrape_and_save(url: str, filename: str = "scraped_data.txt") -> str:
    """
    Scrape content from a webpage and save it to a text file.
    
    Args:
        url: The URL to scrape
        filename: Name of the file to save (default: scraped_data.txt)
    
    Returns:
        Success message with filename and character count
    """
    import requests
    try:
        # Scrape the webpage
        scraped_content = scrape_webpage(url)
        
        if scraped_content.startswith("Error:"):
            return scraped_content
        
        # Create scraped_data directory if it doesn't exist
        save_dir = Path("scraped_data")
        save_dir.mkdir(exist_ok=True)
        
        # Clean filename
        safe_filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        if not safe_filename.endswith('.txt'):
            safe_filename += '.txt'
        
        file_path = save_dir / safe_filename
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"URL: {url}\n")
            f.write(f"Scraped at: {requests.utils.formatdate(localtime=True)}\n")
            f.write("=" * 80 + "\n\n")
            f.write(scraped_content)
        
        return (
            f"✅ Successfully scraped and saved!\n"
            f"📁 File: {file_path}\n"
            f"📊 Characters: {len(scraped_content):,}\n"
            f"📝 Lines: {len(scraped_content.splitlines()):,}"
        )
    
    except Exception as e:
        return f"Error saving scraped data: {str(e)}"

TOOLS = [web_search, calculator, scrape_webpage, scrape_and_save]

# =========================================================
# 3. CREATE AGENT
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=800
)

agent = create_agent(
    llm,
    tools=TOOLS,
    system_prompt="You are a helpful assistant with access to web search, calculator, and web scraping tools. Use web scraping to extract detailed content from URLs. Keep responses concise and informative."
)

# Create direct LLM for general queries (no tools)
direct_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=2000,
    streaming=True
)

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

def sanitize_text(text: Any) -> str:
    """Convert text to safe string for JSON storage."""
    if text is None:
        return ""
    
    text = str(text)
    text = text.replace('\x00', '')
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
    return text.strip()

def create_summary(text: str, max_length: int = 500) -> str:
    """
    Create an intelligent summary of the response.
    Preserves key information within the character limit.
    """
    if not text or len(text) <= max_length:
        return text
    
    # Split into sentences
    sentences = re.split(r'[.!?]+\s+', text)
    
    summary = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Check if adding this sentence would exceed limit
        if len(summary) + len(sentence) + 2 <= max_length:
            summary += sentence + ". "
        else:
            # Add partial sentence if there's room
            remaining = max_length - len(summary)
            if remaining > 50:  # Only add if meaningful space left
                summary += sentence[:remaining-3] + "..."
            break
    
    return summary.strip()

def read_file_content(file_path: str) -> str:
    """
    Read file content with multiple encoding attempts.
    Handles text files robustly.
    """
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")
    
    # If all encodings fail, read as binary and decode with errors ignored
    try:
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    except Exception as e:
        raise Exception(f"Failed to read file with any encoding: {str(e)}")

async def process_uploaded_file(file, streaming_rag):
    """Process uploaded file and add to RAG."""
    print(f"📥 Processing file: {file.name}")
    
    try:
        # Read file content
        if hasattr(file, 'path') and file.path:
            text = read_file_content(file.path)
        elif hasattr(file, 'content'):
            text = file.content.decode("utf-8", errors='ignore')
        else:
            raise Exception("Cannot access file content")
        
        # Add to RAG
        streaming_rag.add_documents(
            documents=[text],
            metadata=[{"source": file.name}]
        )
        
        print(f"✅ Document '{file.name}' indexed successfully")
        print(f"📊 Characters: {len(text):,}")
        print(f"🧠 Total chunks: {streaming_rag.vector_store.index.ntotal}")
        
        return True, text, None
    
    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")
        return False, None, str(e)

# =========================================================
# 5. CHAINLIT INTERFACE
# =========================================================

@cl.on_chat_start
async def on_start():
    """Initialize session with streaming RAG."""
    user_id = cl.user_session.get("id", "guest")
    
    # Initialize streaming RAG
    streaming_rag = StreamingRAG()
    cl.user_session.set("streaming_rag", streaming_rag)
    
    # ✅ Initialize backend session (ASYNC)
    try:
        response = await http_client.post(
            f"{BACKEND_URL}/new_session",
            json={"user_id": user_id}
        )
        session_data = response.json()
        cl.user_session.set("session_id", session_data["id"])
        print(f"✅ Backend session created: {session_data['id']}")
    except Exception as e:
        print(f"❌ Error creating backend session: {e}")
        cl.user_session.set("session_id", None)
    
    await cl.Message(
        content=(
            "👋 Welcome to **Advanced RAG Assistant** with Intelligent Query Routing!\n\n"
            "I can help you with:\n"
            "📚 **Knowledge Base Search** - Upload documents and ask questions\n"
            "🔍 **Web Search** - Search the internet for current information\n"
            "🧮 **Calculator** - Perform mathematical calculations\n"
            "📄 **Web Scraping** - Extract content from URLs\n"
            "💾 **Scrape & Save** - Save scraped content to file\n"
            "💡 **General Knowledge** - Answer questions using AI\n\n"
            "📎 **Upload a document** using the attachment button to get started!\n\n"
            "✨ *Powered by intelligent query routing - I automatically choose the best tool for your question!*"
        )
    ).send()

@cl.on_message
async def on_message(msg: cl.Message):
    """Handle incoming messages with intelligent routing."""
    session_id = cl.user_session.get("session_id")
    streaming_rag = cl.user_session.get("streaming_rag")
    
    if not session_id:
        await cl.Message(content="⚠️ Error: Session not initialized. Backend might be down.").send()
        return
    
    # Check if message has file attachments (elements)
    if msg.elements:
        print(f"📎 Detected {len(msg.elements)} file(s) in message")
        
        for element in msg.elements:
            if isinstance(element, cl.File):
                processing_msg = cl.Message(content=f"⏳ Processing {element.name}...")
                await processing_msg.send()
                
                success, text, error = await process_uploaded_file(element, streaming_rag)
                
                await processing_msg.remove()
                
                if success:
                    await cl.Message(
                        content=(
                            f"✅ **{element.name}** uploaded successfully!\n\n"
                            f"📊 **Size:** {len(text):,} characters\n"
                            f"🧠 **Total chunks in knowledge base:** {streaming_rag.vector_store.index.ntotal}\n\n"
                            "You can now ask questions about this document."
                        )
                    ).send()
                else:
                    await cl.Message(
                        content=f"❌ Failed to process **{element.name}**\n\nError: {error}"
                    ).send()
    
    # Process text message if present
    if msg.content and msg.content.strip():
        # Initialize router if not exists
        if not cl.user_session.get("query_router"):
            router = AdvancedQueryRouter()
            cl.user_session.set("query_router", router)
        else:
            router = cl.user_session.get("query_router")
        
        # Route the query
        has_docs = streaming_rag and streaming_rag.vector_store is not None
        routing_result = await router.route(msg.content, has_documents=has_docs)
        
        # Log routing decision
        print(f"🎯 Routing Decision: {routing_result.datasource}")
        print(f"   Confidence: {routing_result.confidence:.2f}")
        print(f"   Reasoning: {routing_result.reasoning}")
        
        # Send routing info to user (optional - can be removed)
        routing_msg = cl.Message(
            content=f"🔍 Routing to: **{routing_result.datasource}** (confidence: {routing_result.confidence:.0%})"
        )
        await routing_msg.send()
        
        # Small delay to show routing decision
        import asyncio
        await asyncio.sleep(0.5)
        
        # Execute based on routing
        datasource = routing_result.datasource

        if datasource == "rag_vectorstore":
            await handle_streaming_rag(msg.content, streaming_rag, session_id)
        elif datasource == "web_search":
            await handle_agent_with_tool(msg.content, session_id, "web_search")
        elif datasource == "calculator":
            await handle_agent_with_tool(msg.content, session_id, "calculator")
        elif datasource == "web_scraper":
            await handle_agent_with_tool(msg.content, session_id, "scrape_webpage")
        else:  # direct_llm
            await handle_direct_llm(msg.content, session_id)

# =========================================================
# 6. HANDLER FUNCTIONS (✅ ALL ASYNC NOW)
# =========================================================

async def handle_streaming_rag(
    query: str, 
    streaming_rag: StreamingRAG, 
    session_id: str
):
    """Handle streaming RAG response with relevance check."""
    try:
        print("📚 USING RAG PIPELINE")

        # First, check if RAG has relevant information
        chunks, context = await streaming_rag.retrieve_context(query, k=5)
        
        if not chunks:
            # No relevant chunks found - inform user
            await cl.Message(
                content="❌ No relevant information found in uploaded documents.\n\n"
                        "💡 Try rephrasing your question or upload more documents."
            ).send()
            return

        msg = cl.Message(content="")
        await msg.send()
        
        full_response = ""
        
        # Stream the response
        async for chunk in streaming_rag.stream_rag_response(query):
            full_response += chunk
            await msg.stream_token(chunk)
        
        await msg.update()
                
        # Save to backend (✅ ASYNC)
        summary = create_summary(full_response, 500)

        try:
            await http_client.post(
                f"{BACKEND_URL}/save_message",
                params={
                    "session_id": session_id,
                    "question": sanitize_text(query),
                    "answer": sanitize_text(full_response),
                    "summary": sanitize_text(summary)
                }
            )
            print(f"✅ Message saved to backend (Session {session_id})")
        except Exception as e:
            print(f"❌ Error saving message: {e}")
    
    except Exception as e:
        await cl.Message(content=f"Error processing RAG query: {str(e)}").send()
        print(f"RAG Error: {e}")


async def handle_agent_with_tool(query: str, session_id: str, tool_name: str):
    """
    Handle agent response with specific tool emphasis.
    
    Args:
        query: User query
        session_id: Session ID
        tool_name: Name of the tool to emphasize (web_search, calculator, scrape_webpage)
    """
    try:
        print(f"🔧 USING AGENT WITH TOOL: {tool_name}")
        
        # ✅ Get conversation history (ASYNC)
        try:
            history_response = await http_client.get(
                f"{BACKEND_URL}/messages/{session_id}"
            )
            messages_data = history_response.json()
        except Exception as e:
            print(f"⚠️ Could not fetch history: {e}")
            messages_data = []
        
        history = []
        for m in messages_data:
            history.append({"role": "user", "content": sanitize_text(m.get("question", ""))})
            history.append({"role": "assistant", "content": sanitize_text(m.get("answer", ""))})
        
        # Add system message to emphasize tool usage
        tool_prompts = {
            "web_search": "Use the web_search tool to find current information about this query.",
            "calculator": "Use the calculator tool to solve this mathematical problem.",
            "scrape_webpage": "Use the scrape_webpage tool to extract content from the URL mentioned in the query."
        }
        
        emphasized_query = f"{tool_prompts.get(tool_name, '')} {query}"
        history.append({"role": "user", "content": sanitize_text(emphasized_query)})
        
        # Invoke agent
        response = agent.invoke({"messages": history})
        last_message = response["messages"][-1]
        
        if hasattr(last_message, 'content'):
            ai_reply = sanitize_text(last_message.content)
        elif isinstance(last_message, dict):
            ai_reply = sanitize_text(last_message.get("content", "No response"))
        else:
            ai_reply = sanitize_text(str(last_message))
        
        # ✅ Save to backend (ASYNC)
        try:
            await http_client.post(
                f"{BACKEND_URL}/save_message",
                params={
                    "session_id": session_id,
                    "question": sanitize_text(query),
                    "answer": ai_reply,
                    "summary": create_summary(ai_reply, 500)
                }
            )
            print(f"✅ Message saved to backend (Session {session_id})")
        except Exception as e:
            print(f"❌ Error saving message: {e}")
        
        await cl.Message(content=ai_reply).send()
    
    except Exception as e:
        await cl.Message(content=f"Error processing with {tool_name}: {str(e)}").send()
        print(f"Agent Error: {e}")


async def handle_direct_llm(query: str, session_id: str):
    """
    Handle direct LLM response without tools (streaming).
    
    Args:
        query: User query
        session_id: Session ID
    """
    try:
        print("💡 USING DIRECT LLM (No Tools)")
        
        # ✅ Get conversation history (ASYNC)
        try:
            history_response = await http_client.get(
                f"{BACKEND_URL}/messages/{session_id}"
            )
            messages_data = history_response.json()
            
            history = []
            for m in messages_data[-5:]:  # Last 5 messages for context
                history.append({"role": "user", "content": sanitize_text(m.get("question", ""))})
                history.append({"role": "assistant", "content": sanitize_text(m.get("answer", ""))})
        except Exception as e:
            print(f"⚠️ Could not fetch history: {e}")
            history = []
        
        # Add current query
        history.append({"role": "user", "content": sanitize_text(query)})
        
        # Stream response
        msg = cl.Message(content="")
        await msg.send()
        
        full_response = ""
        
        async for chunk in direct_llm.astream(history):
            if chunk.content:
                full_response += chunk.content
                await msg.stream_token(chunk.content)
        
        await msg.update()
        
        # ✅ Save to backend (ASYNC)
        summary = create_summary(full_response, 500)
        
        try:
            await http_client.post(
                f"{BACKEND_URL}/save_message",
                params={
                    "session_id": session_id,
                    "question": sanitize_text(query),
                    "answer": sanitize_text(full_response),
                    "summary": sanitize_text(summary)
                }
            )
            print(f"✅ Message saved to backend (Session {session_id})")
        except Exception as e:
            print(f"❌ Error saving message: {e}")
    
    except Exception as e:
        await cl.Message(content=f"Error processing direct LLM query: {str(e)}").send()
        print(f"Direct LLM Error: {e}")


if __name__ == "__main__":
    print("🚀 Advanced RAG App with Query Routing ready!")
    print("Run with: chainlit run app.py")