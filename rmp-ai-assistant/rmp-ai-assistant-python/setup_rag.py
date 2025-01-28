from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
import requests
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists, if not create it
if "rag" not in pc.list_indexes().names():
    pc.create_index(
        name="rag",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

data = json.load(open("reviews.json"))

processed_data = []
api_key = os.getenv("NEXT_PUBLIC_OPENROUTER_API_KEY")
api_url = "https://api.openrouter.ai/v1/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Create embeddings for each review
for review in data["reviews"]:
    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "input": review['review']
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()  # Ensure we catch any HTTP errors
    embedding = response.json()["data"][0]["embedding"]
    processed_data.append(
        {
            "values": embedding,
            "id": review["professor"],
            "metadata": {
                "review": review["review"],
                "subject": review["subject"],
                "stars": review["stars"],
            }
        }
    )

# Insert the embeddings into the Pinecone index
index = pc.Index("rag")
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Display the index stats
print(index.describe_index_stats())