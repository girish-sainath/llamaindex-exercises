"""
This module defines the ModelFactory class, which is responsible for
creating instances of different models based on the provided model type.
"""
# pylint: disable=invalid-name

import os
from typing import Optional

from dotenv import load_dotenv

from llama_index.llms.litellm import LiteLLM
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama

from src.models.ModelInfo import ModelInfo


load_dotenv()


MODEL_MATRIX = {
    'default': {
        'litellm': ModelInfo.LITELLM_DEFAULT_MODEL.value,
        'ollama': ModelInfo.OLLAMA_DEFAULT_MODEL.value,
        'direct': ModelInfo.DIRECT_DEFAULT_MODEL.value,
    },
    'advanced': {
        'litellm': ModelInfo.LITELLM_ADVANCED_MODEL.value,
        'ollama': ModelInfo.OLLAMA_ADVANCED_MODEL.value,
        'direct': ModelInfo.DIRECT_ADVANCED_MODEL.value,
    },
    'basic': {
        'litellm': ModelInfo.LITELLM_BASIC_MODEL.value,
        'ollama': ModelInfo.OLLAMA_BASIC_MODEL.value,
        'direct': ModelInfo.DIRECT_BASIC_MODEL.value,
    },
}

ACCESS_ALIAS = {
    'litellm': 'litellm',
    'ollama': 'ollama',
}


class ModelFactory:  # pylint: disable=too-few-public-methods
    """
    A factory class for creating model instances based on the provided model name.
    """

    @staticmethod
    def _resolve_model_name(model_type: str, model_access: str) -> str:
        """Return the model name for the given type and access mode."""
        try:
            models_by_access = MODEL_MATRIX[model_type]
        except KeyError as exc:
            raise ValueError(f"Unknown model type: {model_type}") from exc

        access_key = ACCESS_ALIAS.get(model_access, 'direct')
        return models_by_access[access_key]

    @staticmethod
    def _resolve_temperature(model_type: str, temperature: Optional[float]) -> Optional[float]:
        """Apply default temperatures for model types that define one."""
        if model_type == 'default':
            return temperature or ModelInfo.DEFAULT_MODEL_TEMPERATURE.value
        if model_type == 'basic':
            return temperature or ModelInfo.BASIC_MODEL_TEMPERATURE.value
        return temperature

    @staticmethod
    def _build_client(model_access: str, model: str, temperature: Optional[float]):
        """Instantiate the concrete LLM client based on access mode."""
        if model_access == 'litellm':
            print('Using LiteLLM model:', model)
            return LiteLLM(model=model, temperature=temperature)
        if model_access == 'ollama':
            print('Using Ollama model:', model)
            return Ollama(model=model, temperature=temperature)
        print('Using Direct model:', model)
        return Anthropic(model=model, temperature=temperature)

    @staticmethod
    def create_model(
            model_type: str,
            temperature: Optional[float] = None,
            model_access: Optional[str] = None,
    ):
        """
        Create and return an instance of the specified model.

        Args:
            :param model_type: The type of the model.
            :param temperature: Optional temperature setting for the model.
            :param model_access: Optional model access type ('litellm' or 'direct').
        """

        access = model_access or os.getenv('MODEL_ACCESS', 'litellm')
        model = ModelFactory._resolve_model_name(model_type=model_type, model_access=access)
        resolved_temperature = ModelFactory._resolve_temperature(model_type=model_type, temperature=temperature)
        return ModelFactory._build_client(
            model_access=access,
            model=model,
            temperature=resolved_temperature,
        )
