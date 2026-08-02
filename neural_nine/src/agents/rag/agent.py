from dotenv import load_dotenv

from llama_index.llms.litellm import LiteLLM
# from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import BaseWorkflowAgent, ReActAgent

# from src.tools.rag_default.tool import ask_question
from src.tools.rag_faiss.tool import ask_question


load_dotenv()


def get_agent() -> BaseWorkflowAgent:
    """
    Returns a BaseWorkflowAgent that can answer questions about the favorite color.
    :return: A BaseWorkflowAgent instance.
    """
    agent = ReActAgent(
        llm = LiteLLM(
            model='sap/anthropic--claude-4.6-sonnet',
        ),
        # llm = Ollama(
        #     model='qwen3:0.6b',
        # ),
        tools=[ask_question],
        system_prompt='You are a helpful assistant, when you are asked a question, you will use the ask_question tool to answer it.',
    )
    return agent


__all__ = [
    'get_agent',
]
