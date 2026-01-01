"""
LLM Configuration for RedSea GPT - Groq API Integration

Uses Groq API with Llama models for ultra-fast, production-quality responses.
"""

import os
from typing import Optional, List, Any
from pathlib import Path
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from langchain_core.callbacks import CallbackManagerForLLMRun

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars


class GroqLLM(BaseLLM):
    
    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 4096 

    def __init__(
        self,
        api_key: str = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 4096, 
        **kwargs
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set GROQ_API_KEY "
                "environment variable or pass api_key parameter.\n"
                "Get your API key at: https://console.groq.com/keys"
            )

        super().__init__(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "groq"

    def _generate(
        self,
        prompts: List[str],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        
        try:
            import requests

            generations = []
            for prompt in prompts:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    **kwargs
                }

                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()

                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    generations.append([Generation(text=text)])
                else:
                    raise ValueError(f"Unexpected response format: {result}")

            return LLMResult(generations=generations)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling Groq API: {e}")

    @property
    def _identifying_params(self) -> dict:
        """Return identifying parameters for caching."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def create_llm(
    api_key: str = None,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3,
    max_tokens: int = 4096,  # Increased for more comprehensive answers
) -> BaseLLM:
    
    return GroqLLM(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
