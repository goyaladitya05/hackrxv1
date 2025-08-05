# === app/services/vectorstore.py ===
import os
from pinecone import Pinecone, ServerlessSpec

class PineconeVectorStore:
    def __init__(self, index_name: str, dim: int):
        self.index_name = index_name
        self.dim = dim
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust region as needed
            )

        self.index = self.pc.Index(self.index_name)

    def add(self, vectors: list[tuple[str, list[float], dict]]):
        self.index.upsert(vectors=vectors)

    def search(self, query_vector: list[float], top_k: int = 5):
        results = self.index.query(vector=query_vector, top_k=top_k, include_metadata=True)
        return [(match["metadata"]["text"], match["score"]) for match in results["matches"]]
