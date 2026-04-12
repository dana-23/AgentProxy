from google import genai
from google.genai import types

from pydantic import BaseModel, Field
from typing import List
import yaml
from pathlib import Path
from dotenv import load_dotenv

from agentproxy.config.settings import get_settings

prompts_path = Path(__file__).parent.parent / Path("prompts.yaml")

load_dotenv()


class Entity(BaseModel):
    type: str = Field(
        description="Kind of entity (date, period, amount, vendor, category, count, recipient, subject)"
    )
    value: str = Field(description="Normalized value")
    raw: str = Field(description="Original text from user's message")


class ProxyIntent(BaseModel):
    intent: str = Field(description="Short verb-noun phrase describing user's intent")
    entities: List[Entity] = Field(
        default_factory=list, description="Extracted entities from the query"
    )


class Perceive:

    name: str = "perceive"

    def __init__(self) -> None:
        self.client = genai.Client()
        self.settings = get_settings()
        with open(prompts_path, "r") as file:
            data = yaml.safe_load(file)
        self.system_prompt = data.get(f"{self.name}", {}).get("system")

    def __call__(self, prompt: str):
        response = self.client.models.generate_content(
            model=self.settings.default_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                response_mime_type="application/json",
                response_schema=ProxyIntent.model_json_schema(),
            ),
        )

        return response.parsed


if __name__ == "__main__":
    perceive = Perceive()
    result = perceive("What did I buy this week?")
    print(result)
