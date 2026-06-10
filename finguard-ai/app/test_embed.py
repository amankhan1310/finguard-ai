from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

print("API Key Found:", os.getenv("GOOGLE_API_KEY") is not None)

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

print("Creating embedding...")

result = embeddings.embed_query("Hello world")

print("Success!")
print(f"Vector Length: {len(result)}")