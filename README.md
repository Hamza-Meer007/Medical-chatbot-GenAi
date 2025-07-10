# 🏥 Medical Chatbot GenAI

A sophisticated medical chatbot built with **RAG (Retrieval-Augmented Generation)** architecture using **Pinecone vector database**, **Groq's open-source LLaMA model**, **HuggingFace embeddings**, and **Flask** backend. The project is optimized for performance using **UV** package manager and features a modular, production-ready architecture with comprehensive logging.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green)
![LangChain](https://img.shields.io/badge/LangChain-0.3.26-orange)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector--DB-purple)
![Groq](https://img.shields.io/badge/Groq-LLaMA--3.3--70B-red)

## 🚀 Features

- **RAG Architecture**: Retrieval-Augmented Generation for accurate medical responses
- **Vector Database**: Pinecone for efficient similarity search and document retrieval
- **Open Source LLM**: Groq's LLaMA 3.3 70B model for high-quality text generation
- **Modern Embeddings**: HuggingFace sentence-transformers for semantic understanding
- **Fast Package Management**: UV for lightning-fast dependency resolution
- **Comprehensive Logging**: Structured logging with file rotation and multiple levels
- **Modular Architecture**: Clean, maintainable code structure
- **Production Ready**: Error handling, input validation, and monitoring
- **Web Interface**: Modern Flask-based chat interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Flask App     │───▶│   RAG Chain     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                              ┌─────────────────────────┼─────────────────────────┐
                              │                         ▼                         │
                              │               ┌─────────────────┐                 │
                              │               │   Retriever     │                 │
                              │               └─────────────────┘                 │
                              │                         │                         │
                              ▼                         ▼                         ▼
                    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                    │   Groq LLaMA    │    │   Pinecone      │    │  HuggingFace    │
                    │   3.3-70B       │    │   Vector DB     │    │  Embeddings     │
                    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python 3.8+**
- **UV package manager** (for fast dependency management)
- **Pinecone API Key**
- **Groq API Key**
- **Git**

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Medical-chatbot-GenAi.git
cd Medical-chatbot-GenAi
```

### 2. Install UV (if not already installed)

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Set Up Virtual Environment & Install Dependencies

```bash
# Initialize uv and install dependencies
uv init
uv add -r requirements.txt

```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here

# Groq Configuration  
GROK_API_KEY=your_groq_api_key_here

# Optional: Logging Level
LOG_LEVEL=INFO
```

### 5. Prepare Medical Documents

1. Place your medical PDF documents in the `Data/` directory
2. Run the indexing script to create the vector database:

```bash
python store_index.py
```

### 6. Start the Application

```bash
python app.py
```

The application will be available at: `http://localhost:8080`

## 📁 Project Structure

```
Medical-chatbot-GenAi/
├── 📁 Data/                    # Medical PDF documents
│   └── Medical_book.pdf
├── 📁 logs/                    # Application logs (auto-generated)
│   ├── README.md
│   └── medical_chatbot_YYYYMMDD.log
├── 📁 research/                # Research notebooks
│   └── trials.ipynb
├── 📁 src/                     # Source code modules
│   ├── __init__.py
│   ├── helper.py              # Document processing utilities
│   ├── logger.py              # Centralized logging configuration
│   └── prompt.py              # System prompts
├── 📁 static/                  # CSS and static files
│   └── style.css
├── 📁 templates/               # HTML templates
│   └── chat.html
├── app.py                     # Main Flask application
├── store_index.py             # Vector database creation
├── template.py                # Project structure generator
├── requirements.txt           # Python dependencies
├── pyproject.toml            # UV configuration
├── uv.lock                   # UV lock file
├── setup.py                  # Package setup
└── README.md                 # This file
```

## 🔧 Configuration

### API Keys Setup

#### 1. Pinecone API Key

1. Visit [Pinecone](https://www.pinecone.io/)
2. Create an account and project
3. Get your API key from the dashboard
4. Add to `.env` file

#### 2. Groq API Key

1. Visit [Groq](https://console.groq.com/)
2. Create an account
3. Generate an API key
4. Add to `.env` file

### Vector Database Configuration

The system uses Pinecone with the following configuration:

- **Index Name**: `medical-bot`
- **Metric**: Cosine similarity
- **Dimension**: 384 (sentence-transformers/all-MiniLM-L6-v2)
- **Cloud**: AWS
- **Region**: us-east-1

## 🎯 Usage

### Web Interface

1. Start the application: `python app.py`
2. Open browser: `http://localhost:8080`
3. Type your medical question
4. Get AI-powered responses based on your documents

### API Endpoints

#### Chat Endpoint

```http
POST /get
Content-Type: application/x-www-form-urlencoded

msg=What are the symptoms of diabetes?
```

Response:

```json
"Based on medical literature, diabetes symptoms include increased thirst, frequent urination, and unexplained weight loss. Early detection is crucial for proper management."
```

## 📊 Logging

The application features comprehensive logging:

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General application flow
- **WARNING**: Unexpected situations
- **ERROR**: Error conditions
- **CRITICAL**: Serious errors

### Log Locations

- **Console**: Colored output for development
- **File**: `logs/medical_chatbot_YYYYMMDD.log`
- **Rotation**: 10MB max size, 5 backup files

### Log Format

```
2025-07-10 14:30:15 | module_name | INFO | Processing user query: What is diabetes?
2025-07-10 14:30:16 | app | INFO | Generated response length: 245 characters
```

## ⚙️ Advanced Configuration

### Model Parameters

You can customize the LLM behavior in `app.py`:

```python
llm = ChatGroq(
    groq_api_key=GROK_API_KEY,
    model="llama-3.3-70b-versatile",  # Model selection
    temperature=0.3,                   # Creativity level (0-1)
)
```

### Document Processing

Customize chunk processing in `src/helper.py`:

```python
def split_documents(extracted_data, chunk_size=500, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,      # Adjust for longer/shorter chunks
        chunk_overlap=chunk_overlap  # Overlap between chunks
    )
```

### Retrieval Settings

Modify search parameters in `app.py`:

```python
retriever = doc_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Number of documents to retrieve
)
```

## 🔍 Monitoring & Debugging

### Application Health

Check application logs:

```bash
tail -f logs/medical_chatbot_$(date +%Y%m%d).log
```

### Performance Monitoring

The application logs execution times for key operations:

- Document loading
- Embedding generation
- Vector search
- Response generation

### Debug Mode

Enable debug logging:

```python
# In your .env file
LOG_LEVEL=DEBUG
```

## 🛠️ Development

### Adding New Features

1. **New endpoints**: Add routes in `app.py`
2. **Document processors**: Extend `src/helper.py`
3. **Custom prompts**: Modify `src/prompt.py`
4. **Logging**: Use `from src.logger import get_logger`

### Testing

```bash
# Test document loading
python -c "from src.helper import load_pdf; docs = load_pdf('Data/'); print(f'Loaded {len(docs)} documents')"

# Test embeddings
python -c "from src.helper import download_embeddings; emb = download_embeddings(); print('Embeddings working')"

# Test logging
python -c "from src.logger import get_logger; get_logger('test').info('Test message')"
```

## 🚀 Deployment

### Production Deployment

1. **Environment Variables**: Set in production environment
2. **Logging**: Ensure log directory has write permissions
3. **Security**: Use environment variables for sensitive data
4. **Performance**: Consider using Gunicorn for production

```bash
# Install Gunicorn
uv pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "app.py"]
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall dependencies
uv pip install -r requirements.txt
```

#### 2. API Key Issues

```bash
# Verify environment variables
python -c "import os; print('Pinecone:', bool(os.getenv('PINECONE_API_KEY'))); print('Groq:', bool(os.getenv('GROK_API_KEY')))"
```

#### 3. Vector Database Issues

```bash
# Recreate index
python store_index.py
```

#### 4. Logging Issues

- Check `logs/` directory permissions
- Ensure disk space available
- Verify log rotation settings

## 📦 Dependencies

### Core Dependencies

- **flask==3.1.1**: Web framework
- **langchain==0.3.26**: LLM framework
- **langchain-pinecone==0.2.8**: Pinecone integration
- **langchain-groq**: Groq LLM integration
- **sentence-transformers==4.1.0**: Embeddings
- **pypdf==5.6.1**: PDF processing
- **python-dotenv==1.1.0**: Environment management

### Development Dependencies

- **ipykernel**: Jupyter notebook support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add comprehensive logging to new features
- Include error handling
- Update documentation
- Test thoroughly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the excellent RAG framework
- **Pinecone**: For scalable vector database
- **Groq**: For fast LLM inference
- **HuggingFace**: For state-of-the-art embeddings
- **UV**: For blazing-fast package management

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Medical-chatbot-GenAi/issues)
- **Documentation**: See this README and inline code documentation
- **Logs**: Check `logs/` directory for detailed application logs

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Voice interface integration
- [ ] Advanced medical entity recognition
- [ ] Integration with medical databases
- [ ] Enhanced UI/UX with React frontend
- [ ] API rate limiting and authentication
- [ ] Containerized deployment with Docker
- [ ] Automated testing suite
- [ ] Performance metrics dashboard
- [ ] Medical knowledge graph integration

---

**Built with ❤️ for better healthcare through AI**
