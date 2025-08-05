# === app/services/logic.py ===
from app.services.embeddings import EmbeddingService
from app.services.vectorstore import PineconeVectorStore
from app.services.gemini_client import invoke_gemini

embedding_model = EmbeddingService()
embedding_dim = embedding_model.get_embedding_dimension()

index_name = "hackrx-index"
vector_store = PineconeVectorStore(index_name=index_name, dim=embedding_dim)

def retrieve_and_respond(text_chunks: list[str], questions: list[str]) -> dict:
    chunk_vectors = embedding_model.embed(text_chunks)
    vectors_to_upsert = [(str(i), vec.tolist(), {"text": chunk}) for i, (vec, chunk) in enumerate(zip(chunk_vectors, text_chunks))]
    vector_store.add(vectors_to_upsert)

    results = []
    for q in questions:
        q_vec = embedding_model.embed([q])[0].tolist()
        top_chunks = vector_store.search(q_vec, top_k=5)
        context = "\n".join([c for c, _ in top_chunks])
        prompt = f"You are a helpful assistant. Given the context below, answer the user's question in a single concise sentence. You may use commas but do not use bullet points or new lines.\n\nContext:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        response = invoke_gemini(prompt)
        results.append({"question": q, "answer": response})

    return restructure_response(results)

def restructure_response(response) -> dict:
    if isinstance(response, dict):
        response = response.get("answers", [])

    answers = [
        item["answer"].strip()
        for item in response
        if item.get("answer", "").strip()
    ]
    return {"answers": answers}
