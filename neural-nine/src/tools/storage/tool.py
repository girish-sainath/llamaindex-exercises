from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding


load_dotenv()


Settings.llm = LiteLLM(model='sap/anthropic--claude-4.6-sonnet')
Settings.embed_model = LiteLLMEmbedding(model_name='sap/text-embedding-3-large')

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

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
