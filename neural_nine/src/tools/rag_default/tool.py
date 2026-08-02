from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding

load_dotenv()


Settings.llm = LiteLLM(model='sap/anthropic--claude-4.6-sonnet')
Settings.embed_model = LiteLLMEmbedding(model_name='sap/text-embedding-3-large')

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()


def ask_question(question: str) -> str:
    """
    Ask a question to the query engine and return the response.
    :param question: The question to ask.
    :return: The response from the query engine.
    """
    response = query_engine.query(question)
    index.storage_context.persist('storage')
    return str(response)

__all__ = [
    'ask_question',
]
