"""
System prompts for the Medical Chatbot GenAI application.
This module contains the system prompt used by the RAG chain for generating responses.
"""

from .logger import get_logger

# Initialize logger
logger = get_logger(__name__)

logger.info("Loading system prompt for medical chatbot")

system_prompt = (
'''
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know and give the user suggestions to ask questions like medical diseases. Use three sentences maximum and keep the 
answer concise. no preamble, just the answer.
{context}

'''
)

logger.info("System prompt loaded successfully")