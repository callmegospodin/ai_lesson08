import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-4o-mini"

llm = OpenRouter(
    api_key=OPENROUTER_API_KEY,
    model=OPENROUTER_MODEL,
    max_tokens=1024,
    temperature=0.7,
    additional_headers={
        "HTTP-Referer": "https://your-site.com",
        "X-Title": "Your App Name"
    }
)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("Що таке Кінорецензія?")
print(response)