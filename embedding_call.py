from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("KKB_API")

# Initialize client with your custom base URL
client = OpenAI(base_url="https://mia.csp.kloudeks.com/v1", api_key=api_key)

# Define your query
query_text = "Naber, nasılsın?"

# Get the embedding
resp = client.embeddings.create(
   model="qwen3-embedding-8b",
   input=query_text,
   encoding_format="float" # Optional: defaults to float
)

# Extract the vector
embedding_vector = resp.data[0].embedding

print(f"Embedding length: {len(embedding_vector)}")
print(f"First 5 values: {embedding_vector[:5]}")