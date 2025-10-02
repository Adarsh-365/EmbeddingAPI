import os
import asyncio
import logging
import logging.config
from pathlib import Path
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from openaikey import ForgeAPIClient
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import requests

# Configuration
WORKING_DIR = "./knowledge_graph_storage2"
MARKDOWN_DIR = "./uploaded_docs"  # Directory containing markdown files

def configure_logging():
    """Configure logging for the application"""
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "knowledge_graph_creator.log"))
    print(f"\nKnowledge Graph Creator log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
        },
        "handlers": {
            "console": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr"
            },
            "file": {
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            }
        }
    })

    logger.setLevel(logging.INFO)
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")

def create_directories():
    """Create necessary working directory"""
    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"📁 Created working directory: {WORKING_DIR}")

# Global model instance for reuse
_embedding_model = None

def get_embedding_model():
    """Get or create global embedding model instance for reuse"""
    global _embedding_model
    if _embedding_model is None:
        print("🚀 Loading SentenceTransformer model (one-time setup)...")
        try:
            # Configure proxy for Hugging Face
            os.environ['HTTP_PROXY'] = 'http://proxy-dmz.intel.com:912'
            os.environ['HTTPS_PROXY'] = 'http://proxy-dmz.intel.com:912'
            
            # Use a faster, smaller model for better performance
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _embedding_model = None
    return _embedding_model

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Optimized embedding function using SentenceTransformer with parallel processing"""
    try:
        model = get_embedding_model()
        if model is None:
            print("Using random embeddings as fallback")
            return np.random.randn(len(texts), 384).astype(np.float32)
        
        # Process in parallel batches for better performance
        import time
        start_time = time.time()
        
        # For large batches, process in chunks to avoid memory issues
        batch_size = 32  # Optimal batch size for most systems
        all_embeddings = []
        
        if len(texts) <= batch_size:
            # Small batch - process all at once
            embeddings = model.encode(texts, 
                                    convert_to_numpy=True, 
                                    batch_size=batch_size,
                                    show_progress_bar=True if len(texts) > 5 else False)
            all_embeddings = embeddings
        else:
            # Large batch - process in chunks with progress
            print(f"📊 Processing {len(texts)} texts in batches of {batch_size}...")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = model.encode(batch_texts, 
                                              convert_to_numpy=True, 
                                              batch_size=batch_size,
                                              show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                
                # Progress indicator
                processed = min(i + batch_size, len(texts))
                print(f"  ⚡ Processed {processed}/{len(texts)} texts ({processed/len(texts)*100:.1f}%)")
            
            all_embeddings = np.array(all_embeddings)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Embedding completed: {len(texts)} texts in {processing_time:.2f}s")
        print(f"   📏 Shape: {all_embeddings.shape}")
        print(f"   ⚡ Speed: {len(texts)/processing_time:.1f} texts/second")
        
        return all_embeddings.astype(np.float32)
        
    except Exception as e:
        print(f"❌ Error in embedding function: {e}")
        print("🔄 Using random embeddings as last resort")
        return np.random.randn(len(texts), 384).astype(np.float32)

def get_fresh_llm():
    """Get a fresh LLM instance with updated token"""
    os.environ["OPENAI_API_KEY"] = ForgeAPIClient().get_access_token()
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        max_tokens=4000,
        timeout=180,
        max_retries=3,
        base_url="https://apis-internal.intel.com/generativeaiinference/v4", 
        openai_proxy='http://proxy-dmz.intel.com:912'
    )

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    """Enhanced LLM function for knowledge graph extraction"""
    
    def sync_llm_call():
        try:
            llm = get_fresh_llm()
            
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            if history_messages:
                messages.extend(history_messages)
            
            messages.append({"role": "user", "content": prompt})
            
            # Make the call
            response = llm.invoke(messages)
            content = response.content
            
            # Enhanced handling for entity extraction
            if ("entity" in prompt.lower() and "relationship" in prompt.lower() and 
                ("##" in prompt or "entity_name" in prompt or "source_entity" in prompt)):
                
                if not content or not ("entity" in content or "relationship" in content):
                    print(f"Warning: LLM returned empty or invalid extraction format")
                    return ""
                
            return content
            
        except Exception as e:
            print(f"Error in LLM call: {e}")
            if ("entity" in prompt.lower() and "relationship" in prompt.lower()):
                return ""
            return "I apologize, but I encountered an error processing your request."
    
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sync_llm_call)
        return result
        
    except Exception as e:
        print(f"Error in async LLM wrapper: {e}")
        return ""



def read_markdown_file(file_path: Path) -> str:
    """Read and return the content of a markdown file with size optimization"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        file_size_mb = len(content) / (1024 * 1024)
        print(f"📄 Read {file_path.name}: {len(content):,} chars ({file_size_mb:.2f} MB)")
        
        # Warn about very large files
        if file_size_mb > 10:
            print(f"⚠️  Large file detected ({file_size_mb:.1f} MB) - processing may take time")
        
        return content
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""

def chunk_large_content(content: str, max_chunk_size: int = 8000) -> list[str]:
    """Split large content into smaller chunks for better processing"""
    if len(content) <= max_chunk_size:
        return [content]
    
    # Split by paragraphs first, then by sentences if needed
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"📝 Split large content into {len(chunks)} chunks (avg {len(content)//len(chunks):,} chars each)")
    return chunks

async def initialize_rag() -> LightRAG:
    """Initialize the LightRAG system with optimized settings"""
    print("🔧 Initializing LightRAG system...")
    
    # Pre-load the embedding model and determine dimension
    try:
        print("🧠 Testing embedding function...")
        test_texts = ["test sentence for dimension detection"]
        test_embedding = await embedding_func(test_texts)
        embedding_dim = test_embedding.shape[1]
        print(f"✅ Embedding dimension detected: {embedding_dim}")
    except Exception as e:
        print(f"⚠️  Error testing embedding function: {e}")
        embedding_dim = 384  # Default for all-MiniLM-L6-v2
        print(f"🔄 Using default embedding dimension: {embedding_dim}")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=embedding_func,
        ),
        llm_model_func=llm_model_func,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    print("✅ LightRAG system initialized successfully!")
    return rag

async def process_markdown_files_to_graph(file_path: str = None):
    """Process a single markdown file and create knowledge graph"""
    
    if not file_path:
        print("❌ No file path provided!")
        return None
    
    try:
        # Convert to Path object and validate
        md_file = Path(file_path)
        if not md_file.exists():
            print(f"❌ File does not exist: {file_path}")
            return None
        
        if not md_file.suffix.lower() == '.md':
            print(f"❌ File is not a markdown file: {file_path}")
            return None
        
        print(f"📄 Processing markdown file: {md_file.name}")
        
        # Initialize RAG system
        print("🚀 Initializing LightRAG system...")
        rag = await initialize_rag()
        
        total_chunks_processed = 0
        
        print(f"\n{'='*60}")
        print(f"📄 Processing file: {md_file.name}")
        print(f"{'='*60}")
        
        # Read the markdown content
        content = read_markdown_file(md_file)
        
        if content:
            # Check if content is very large and needs chunking
            content_size_mb = len(content) / (1024 * 1024)
            
            if content_size_mb > 5:  # Large file (>5MB)
                print(f"🔄 Large file detected ({content_size_mb:.1f} MB) - processing in chunks...")
                chunks = chunk_large_content(content, max_chunk_size=6000)
                
                for j, chunk in enumerate(chunks, 1):
                    print(f"  📝 Processing chunk {j}/{len(chunks)}...")
                    try:
                        await rag.ainsert(chunk)
                        total_chunks_processed += 1
                        print(f"    ✅ Chunk {j} processed successfully")
                    except Exception as e:
                        print(f"    ❌ Error processing chunk {j}: {e}")
                
                print(f"✅ Completed {md_file.name} ({len(chunks)} chunks)")
            else:
                # Normal size file - process as whole
                try:
                    print(f"📝 Processing file content...")
                    await rag.ainsert(content)
                    total_chunks_processed += 1
                    print(f"✅ Successfully processed {md_file.name}")
                except Exception as e:
                    print(f"❌ Error processing {md_file.name}: {e}")
        else:
            print(f"⚠️  Skipped {md_file.name} due to read error")
            return None
        
        print(f"\n{'='*60}")
        print("🎉 KNOWLEDGE GRAPH CREATION COMPLETED!")
        print(f"📊 Statistics:")
        print(f"   📄 File processed: {md_file.name}")
        print(f"   📝 Total chunks: {total_chunks_processed}")
        print(f"   💾 Graph stored in: {WORKING_DIR}")
        print(f"{'='*60}")
        
        return rag
        
    except Exception as e:
        print(f"❌ Error in processing: {e}")
        import traceback
        traceback.print_exc()
        return None



async def main(file_path: str = None):
    """Main execution function - Creates knowledge graph from a markdown file"""
    # Set up environment
    os.environ["OPENAI_API_KEY"] = ForgeAPIClient().get_access_token()
    
    # If no file path provided, ask user for input
    if not file_path:
        print("📄 Please provide the path to your markdown file:")
        file_path = input("Enter file path: ").strip().strip('"')
    
    if not file_path:
        print("❌ No file path provided!")
        return
    
    try:
        # Create necessary directories
        create_directories()
        
        # Process the markdown file and create knowledge graph
        rag = await process_markdown_files_to_graph(file_path)
        
        if rag:
            print(f"\n{'='*60}")
            print("🎉 KNOWLEDGE GRAPH CREATION SUCCESS!")
            print(f"📊 Your knowledge graph is ready and stored in: {WORKING_DIR}")
            print(f"🔍 You can now use this graph for queries in other scripts")
            print(f"{'='*60}")
            
            # Finalize and cleanup
            await rag.finalize_storages()
            print("✅ Knowledge graph storage finalized successfully!")
        else:
            print("❌ Failed to create knowledge graph")
        
    except Exception as e:
        print(f"❌ An error occurred in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    configure_logging()
    print("🚀 Starting Knowledge Graph Creator from Markdown File")
    print(f" Knowledge graph will be stored in: {WORKING_DIR}")
    
    # You can also pass a file path directly
    # Example: asyncio.run(main("path/to/your/file.md"))
    asyncio.run(main(file_path="C:\\Users\\atayde\\OneDrive - Intel Corporation\\Desktop\\Work\\Graphrag\\LightRAG\\examples\\uploaded_docs\\report.md"))
    print("\n✅ Done!")
