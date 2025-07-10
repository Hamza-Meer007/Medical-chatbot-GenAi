"""
Pinecone Vector Store Index Creation Script
This script loads PDF documents, processes them, and creates a Pinecone vector store index
for the medical chatbot RAG system.
"""

from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
import warnings
from src.helper import load_pdf, split_documents, download_embeddings
from src.logger import get_logger, log_execution_time

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize logger
logger = get_logger(__name__)

@log_execution_time
def create_vector_store():
    """
    Create or update the Pinecone vector store with processed documents.
    
    Raises:
        Exception: If vector store creation fails
    """
    logger.info("Starting Pinecone vector store creation process")
    
    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")
        
        # Get API key
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        logger.info("Pinecone API key retrieved successfully")
        
        # Initialize Pinecone client
        logger.info("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        index_name = "medical-bot"
        logger.info(f"Working with index: {index_name}")
        
        # Check if index exists, create if not
        if not pc.has_index(index_name):
            logger.info(f"Index {index_name} does not exist. Creating new index...")
            pc.create_index(
                name=index_name,
                metric="cosine",  # Metric for similarity search
                dimension=384,  # Dimension of the embeddings
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                )
            )
            logger.info(f"Successfully created index: {index_name}")
        else:
            logger.info(f"Index {index_name} already exists")
        
        # Load and process documents
        logger.info("Loading PDF documents...")
        extracted_data = load_pdf('Data/')
        
        logger.info("Splitting documents into chunks...")
        text_chunks = split_documents(extracted_data)
        
        logger.info("Initializing embeddings model...")
        embeddings = download_embeddings()
        
        # Create vector store
        logger.info(f"Creating vector store with {len(text_chunks)} document chunks...")
        vector_store = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        
        logger.info("Vector store created successfully!")
        logger.info(f"Indexed {len(text_chunks)} document chunks in Pinecone")
        
        return vector_store
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise Exception(f"Failed to create vector store: {str(e)}")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("MEDICAL CHATBOT - VECTOR STORE CREATION")
    logger.info("="*60)
    
    try:
        vector_store = create_vector_store()
        logger.info("Vector store creation completed successfully!")
        logger.info("The medical chatbot is ready to use.")
        
    except Exception as e:
        logger.critical(f"Vector store creation failed: {str(e)}")
        exit(1)