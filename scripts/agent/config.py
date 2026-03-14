import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class AgentConfig(BaseModel):
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    yelp_api_key: str = Field(default_factory=lambda: os.getenv("YELP_API_KEY", ""))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    confidence_threshold: float = Field(default=0.65)
    max_credits_per_run: int = Field(default=4000)

    def validate_keys(self):
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set. Please set it in your environment or .env file.")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set. Please set it in your environment or .env file.")

config = AgentConfig()
