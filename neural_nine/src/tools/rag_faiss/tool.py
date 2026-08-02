import faiss
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


load_dotenv()


Settings.llm = LiteLLM(model='sap/anthropic--claude-4.6-sonnet')
Settings.embed_model = LiteLLMEmbedding(model_name='sap/text-embedding-3-large')

documents = SimpleDirectoryReader('data').load_data()
vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(3072))
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=LiteLLMEmbedding(model_name='sap/text-embedding-3-large'),
)

query_engine = index.as_query_engine()


def ask_question(question: str) -> str:
    """
    Ask a question to the query engine and return the response.
    :param question: The question to ask.
    :return: The response from the query engine.
    """
    response = query_engine.query(question)
    return str(response)

__all__ = [
    'ask_question',
]
