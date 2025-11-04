"""
Elasticsearch Index Management
"""
from elasticsearch import Elasticsearch
from config import ElasticConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexManager:
    def __init__(self):
        self.config = ElasticConfig()
        self.es = Elasticsearch([self.config.url], verify_certs=False)
        
        if self.es.ping():
            logger.info("✓ Successfully connected to Elasticsearch")
        else:
            raise ConnectionError("Failed to connect to Elasticsearch")
    
    def create_index(self, index_name: str) -> bool:
        if self.es.indices.exists(index=index_name):
            logger.warning(f"Index '{index_name}' already exists")
            return True
        
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_id": {"type": "keyword"}
                }
            }
        }
        
        try:
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"✓ Created index '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create index: {str(e)}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        try:
            if self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"✓ Deleted index '{index_name}'")
                return True
            return False
        except Exception as e:
            logger.error(f"✗ Failed to delete index: {str(e)}")
            return False
    
    def index_exists(self, index_name: str) -> bool:
        return self.es.indices.exists(index=index_name)
    
    def get_index_stats(self, index_name: str) -> dict:
        try:
            stats = self.es.indices.stats(index=index_name)
            doc_count = stats['indices'][index_name]['total']['docs']['count']
            size = stats['indices'][index_name]['total']['store']['size_in_bytes']
            return {
                'document_count': doc_count,
                'size_bytes': size,
                'size_mb': round(size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"✗ Failed to get index stats: {str(e)}")
            return {}

