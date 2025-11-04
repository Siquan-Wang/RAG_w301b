"""
Configuration file for RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Elasticsearch Configuration
class ElasticConfig:
    """Elasticsearch connection configuration"""
    url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    username = os.getenv('ELASTICSEARCH_USERNAME', 'elastic')
    password = os.getenv('ELASTICSEARCH_PASSWORD', '')
    
    number_of_shards = 1
    number_of_replicas = 0
    vector_dims = 1024
    similarity = "cosine"

# Embedding Service Configuration
EMBEDDING_URL = os.getenv('EMBEDDING_URL', 'http://localhost:8000/v1/embeddings')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'bge-large-zh-v1.5')

# Reranking Service Configuration
RERANK_URL = os.getenv('RERANK_URL', 'http://localhost:8001/rerank')
RERANK_MODEL = os.getenv('RERANK_MODEL', 'bge-reranker-v2-m3')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

# Document Processing Configuration
class DocumentConfig:
    """Document processing configuration"""
    chunk_size = 1024
    chunk_overlap = 100
    batch_size = 25
    extract_images = True
    image_caption_model = 'gpt-4o-mini'
    extract_tables = True
    table_to_markdown = True

# Retrieval Configuration
class RetrievalConfig:
    """Retrieval and search configuration"""
    max_results_per_query = 20
    rrf_k = 60
    num_query_variations = 3
    similarity_threshold = 0.8
    top_k_rerank = 50
    final_top_k = 10

# Response Generation Configuration
class GenerationConfig:
    """Response generation configuration"""
    temperature = 0.7
    max_tokens = 2000
    top_p = 0.9
    system_prompt = """You are a helpful AI assistant that answers questions based on provided documents.
Always cite your sources by referring to the document chunks you used.
If you cannot find relevant information in the provided context, say so clearly.
Provide detailed, accurate, and well-structured responses."""

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'rag_system.log')

