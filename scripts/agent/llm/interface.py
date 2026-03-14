import os
from pydantic import BaseModel
from typing import TypeVar, Type, Optional, Any
from abc import ABC, abstractmethod
from google import genai
from config import config

T = TypeVar('T', bound=BaseModel)

class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate_structured_output(self, prompt: str, schema: Type[T]) -> Optional[T]:
        """Generate a structured output matching the provided Pydantic schema."""
        pass


class GeminiLLM(LLMInterface):
    """Implementation of LLMInterface using the Google Gemini model."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.client = genai.Client(api_key=config.gemini_api_key)

    def generate_structured_output(self, prompt: str, schema: Type[T]) -> Optional[T]:
        try:
            extraction = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema
                }
            )
            return extraction.parsed
        except Exception as e:
            # Raise the exception so the caller can handle/log it.
            raise e

class GroqLLM(LLMInterface):
    """Implementation of LLMInterface using the Groq API for rapid inference."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Please install groq to use GroqLLM: pip install groq")
            
        self.model_name = model_name
        self.client = Groq(api_key=config.groq_api_key)

    def generate_structured_output(self, prompt: str, schema: Type[T]) -> Optional[T]:
        # Groq's json_object mode requires the prompt to instruct it strictly
        import json
        schema_dict = schema.model_json_schema()
        
        system_instructions = (
            "You are a helpful assistant. You must output your answer entirely in JSON format. "
            f"Your output must perfectly match this JSON schema:\n{json.dumps(schema_dict, indent=2)}"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            parsed_json = response.choices[0].message.content
            return schema.model_validate_json(parsed_json)
        except Exception as e:
            raise e
