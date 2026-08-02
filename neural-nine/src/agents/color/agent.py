from dotenv import load_dotenv

from llama_index.core.agent.workflow import BaseWorkflowAgent, ReActAgent

from src.tools.color.tool import get_favorite_color
from src.models.ModelInfo import ModelInfo
from src.models.ModelFactory import ModelFactory

load_dotenv()


def get_agent() -> BaseWorkflowAgent:
    """
    Returns a BaseWorkflowAgent that can answer questions about the favorite color.
    :return: A BaseWorkflowAgent instance.
    """
    agent = ReActAgent(
        llm = ModelFactory.create_model(
            ModelInfo.DEFAULT_MODEL_TYPE.value,
        ),
        tools=[get_favorite_color],
        system_prompt='You are a helpful assistant.',
    )
    return agent


__all__ = [
    'get_agent',
]
