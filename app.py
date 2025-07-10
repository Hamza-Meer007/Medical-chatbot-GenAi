"""
Medical Chatbot GenAI Application
A Flask-based medical chatbot using RAG (Retrieval-Augmented Generation) with Pinecone vector store,
Groq LLM, and HuggingFace embeddings for answering medical questions.
"""

from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import traceback
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from src.helper import download_embeddings
from src.prompt import *
from src.logger import get_logger, log_execution_time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize logger
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
logger.info("Initializing Medical Chatbot GenAI Application")

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded successfully")

try:
    # Initialize embeddings
    logger.info("Initializing embeddings model...")
    embeddings = download_embeddings()
    logger.info("Embeddings model initialized successfully")
    
    # Get API keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    if not GROK_API_KEY:
        raise ValueError("GROK_API_KEY not found in environment variables")
    
    logger.info("API keys retrieved successfully")
    
    # Initialize Pinecone
    logger.info("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    index_name = "medical-bot"
    logger.info(f"Using Pinecone index: {index_name}")
    
    # Connect to existing Pinecone index
    logger.info("Connecting to Pinecone vector store...")
    doc_search = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    logger.info("Successfully connected to Pinecone vector store")
    
    # Create retriever
    retriever = doc_search.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info("Document retriever configured with similarity search (k=3)")
    
    # Initialize Groq LLM
    logger.info("Initializing Groq LLM...")
    llm = ChatGroq(
        groq_api_key=GROK_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )
    logger.info("Groq LLM initialized successfully with model: llama-3.3-70b-versatile")
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    logger.info("Chat prompt template created")
    
    # Create chains
    logger.info("Creating document processing chains...")
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    logger.info("RAG chain created successfully")
    
except Exception as e:
    logger.critical(f"Failed to initialize application components: {str(e)}")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    raise



@app.route("/")
@log_execution_time
def index():
    """Render the main chat interface."""
    logger.info("Serving main chat interface")
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error serving chat interface: {str(e)}")
        return jsonify({"error": "Failed to load chat interface"}), 500


@app.route("/get", methods=["GET", "POST"])
@log_execution_time
def chat():
    """
    Handle chat requests and generate responses using RAG.
    
    Returns:
        str: Generated response from the medical chatbot
    """
    try:
        # Get user message
        msg = request.form.get("msg", "").strip()
        
        if not msg:
            logger.warning("Empty message received from user")
            return "Please enter a valid question."
        
        logger.info(f"Processing user query: {msg[:100]}...")  # Log first 100 chars
        
        # Validate message length
        if len(msg) > 1000:
            logger.warning(f"Message too long ({len(msg)} characters): {msg[:50]}...")
            return "Please keep your question under 1000 characters."
        
        # Generate response using RAG chain
        logger.info("Invoking RAG chain to generate response...")
        response = rag_chain.invoke({"input": msg})
        
        answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
        
        # Log response metrics
        logger.info(f"Generated response length: {len(answer)} characters")
        logger.info(f"Response preview: {answer[:100]}...")
        
        # Log retrieved context info if available
        if "context" in response:
            context_docs = response["context"]
            logger.info(f"Retrieved {len(context_docs)} context documents for response")
        
        return str(answer)
        
    except Exception as e:
        error_msg = f"Error processing chat request: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "I'm sorry, I encountered an error processing your request. Please try again."


@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    logger.warning(f"404 error: {request.url}")
    return jsonify({"error": "Page not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    logger.info("Starting Medical Chatbot Flask application")
    logger.info("Application configuration:")
    logger.info(f"  - Host: 0.0.0.0")
    logger.info(f"  - Port: 8080")
    logger.info(f"  - Debug mode: True")
    logger.info("Application is ready to serve requests")
    
    try:
        app.run(host="0.0.0.0", port=8080, debug=True)
    except Exception as e:
        logger.critical(f"Failed to start Flask application: {str(e)}")
        raise
