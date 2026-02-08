import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    model_name: str = 'openai/gpt-5-nano'
    openai_url: str = os.getenv('OPENAI_URL', 'https://api.vsegpt.ru/v1')

    tavily_api_key: str = os.getenv('TAVILY_API_KEY')
    openai_api_key: str = os.getenv('VSEGPT_API_KEY')


config = Config()
