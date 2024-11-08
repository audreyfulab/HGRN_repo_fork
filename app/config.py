import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file

class Config:
    ATTENTION_HEADS = os.environ['ATTENTION_HEADS']