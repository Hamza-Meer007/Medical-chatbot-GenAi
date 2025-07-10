"""
Project Template Creation Script
This script creates the initial directory structure and files for the Medical Chatbot GenAI project.
"""

import os
from pathlib import Path
import warnings
from src.logger import get_logger

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize logger
logger = get_logger(__name__)

# List of files to create
list_of_files = [
    "src/__init__.py",
    "src/helper.py", 
    "src/prompt.py",
    "src/logger.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb",
    "requirements.txt",
    "static/style.css",
    "templates/chat.html",
    "Data/.gitkeep",
    "logs/.gitkeep",
]

def create_project_structure():
    """Create the project directory structure and files."""
    logger.info("="*60)
    logger.info("MEDICAL CHATBOT - PROJECT TEMPLATE CREATION")
    logger.info("="*60)
    
    created_dirs = set()
    created_files = []
    existing_files = []
    
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)

        # Create directory if it doesn't exist
        if filedir != "" and filedir not in created_dirs:
            os.makedirs(filedir, exist_ok=True)
            created_dirs.add(filedir)
            logger.info(f"Created directory: {filedir}")

        # Create file if it doesn't exist or is empty
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
            with open(filepath, "w") as f:
                if filename == ".gitkeep":
                    f.write("# This file ensures the directory is tracked by git\n")
                elif filename.endswith(".py"):
                    f.write(f'"""\n{filename} - Part of Medical Chatbot GenAI project\n"""\n\n')
                pass
            created_files.append(str(filepath))
            logger.info(f"Created file: {filepath}")
        else:
            existing_files.append(str(filepath))
            logger.info(f"File already exists: {filename}")
    
    # Summary
    logger.info("="*60)
    logger.info("PROJECT TEMPLATE CREATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Directories created: {len(created_dirs)}")
    logger.info(f"Files created: {len(created_files)}")
    logger.info(f"Files already existing: {len(existing_files)}")
    
    if created_dirs:
        logger.info("Created directories:")
        for directory in sorted(created_dirs):
            logger.info(f"  - {directory}")
    
    if created_files:
        logger.info("Created files:")
        for file in created_files:
            logger.info(f"  - {file}")


if __name__ == "__main__":
    try:
        create_project_structure()
        logger.info("Project template creation completed successfully!")
    except Exception as e:
        logger.error(f"Error creating project template: {str(e)}")
        exit(1)