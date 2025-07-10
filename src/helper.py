"""
Helper functions for document processing and embeddings.
This module contains utility functions for loading PDFs, splitting documents, and downloading embeddings.
"""

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from .logger import get_logger, log_execution_time
import os

# Get logger for this module
logger = get_logger(__name__)


@log_execution_time
def load_pdf(path: str):
    """
    Load PDF documents from a directory.
    
    Args:
        path (str): Path to directory containing PDF files
        
    Returns:
        list: List of loaded documents
        
    Raises:
        FileNotFoundError: If the specified path doesn't exist
        Exception: If document loading fails
    """
    logger.info(f"Starting to load PDF documents from path: {path}")
    
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        
        # Count PDF files for logging
        pdf_files = []
        for root, dirs, files in os.walk(path):
            pdf_files.extend([f for f in files if f.lower().endswith('.pdf')])
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        logger.info(f"Successfully loaded {len(documents)} document chunks from {len(pdf_files)} PDF files")
        
        # Log some statistics
        total_chars = sum(len(doc.page_content) for doc in documents)
        logger.info(f"Total content loaded: {total_chars:,} characters")
        
        return documents
        
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading PDF documents: {str(e)}")
        raise Exception(f"Failed to load PDF documents: {str(e)}")


@log_execution_time
def split_documents(extracted_data: list, chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into smaller chunks for processing.
    
    Args:
        extracted_data (list): List of documents to split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        list: List of split document chunks
        
    Raises:
        ValueError: If extracted_data is empty or invalid
        Exception: If document splitting fails
    """
    logger.info(f"Starting document splitting with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    try:
        if not extracted_data:
            raise ValueError("No documents provided for splitting")
        
        logger.info(f"Input: {len(extracted_data)} documents to split")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        split_data = splitter.split_documents(extracted_data)
        
        logger.info(f"Successfully split documents into {len(split_data)} chunks")
        
        # Log chunk statistics
        if split_data:
            avg_chunk_size = sum(len(chunk.page_content) for chunk in split_data) / len(split_data)
            logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
        
        return split_data
        
    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise Exception(f"Failed to split documents: {str(e)}")


@log_execution_time
def download_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Download and initialize embeddings model.
    
    Args:
        model_name (str): Name of the HuggingFace model to use
        
    Returns:
        HuggingFaceEmbeddings: Initialized embeddings object
        
    Raises:
        Exception: If model download or initialization fails
    """
    logger.info(f"Initializing embeddings model: {model_name}")
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embeddings model: {model_name}")
        
        # Test embedding to ensure it's working
        test_text = "This is a test sentence."
        test_embedding = embeddings.embed_query(test_text)
        logger.info(f"Embeddings model working correctly. Embedding dimension: {len(test_embedding)}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Error initializing embeddings model {model_name}: {str(e)}")
        raise Exception(f"Failed to initialize embeddings model: {str(e)}")