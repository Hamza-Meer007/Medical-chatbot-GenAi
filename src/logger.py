"""
Centralized logging configuration for Medical Chatbot GenAI Application.
Provides colored console output, file logging with rotation, and performance monitoring.
"""

import logging
import logging.handlers
import os
import time
import warnings
from datetime import datetime
from functools import wraps
from typing import Any, Callable

# Suppress all warnings from external libraries
warnings.filterwarnings("ignore")

# Suppress specific warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress warnings from specific libraries
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=FutureWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="pinecone")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        return formatted


def setup_logging() -> logging.Logger:
    """
    Set up centralized logging configuration.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create logger
    logger = logging.getLogger("medical_chatbot")
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with rotation
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(logs_dir, f"medical_chatbot_{current_date}.log")
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Set up the main logger if not already done
    main_logger = setup_logging()
    
    # Return a child logger with the specified name
    return logging.getLogger(f"medical_chatbot.{name}")


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func (Callable): Function to be decorated
        
    Returns:
        Callable: Decorated function
    """
    @wraps(func)  # This preserves the original function's name and metadata
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            logger.debug(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed successfully in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
            
    return wrapper


def log_memory_usage():
    """Log current memory usage if psutil is available."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger = get_logger("system")
        logger.debug(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        pass


# Suppress warnings from external libraries at import time
def suppress_external_warnings():
    """Suppress warnings from external libraries."""
    
    # Transformers warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    
    # LangChain warnings
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("langchain_core").setLevel(logging.ERROR)
    
    # Pinecone warnings
    logging.getLogger("pinecone").setLevel(logging.ERROR)
    
    # HTTP/urllib3 warnings
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.ERROR)
    
    # Other common libraries
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Apply warning suppression
suppress_external_warnings()

# Example usage and testing
if __name__ == "__main__":
    # Test the logging setup
    test_logger = get_logger(__name__)
    
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.critical("This is a critical message")
    
    # Test the execution time decorator
    @log_execution_time
    def test_function():
        """Test function for timing."""
        time.sleep(1)
        return "Test completed"
    
    result = test_function()
    test_logger.info(f"Test result: {result}")