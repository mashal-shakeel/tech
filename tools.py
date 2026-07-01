import chromadb
from sentence_transformers import SentenceTransformer

from python_docs import search_python_docs

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection("python_tutorial")


def retrieve_documents(question: str, n_results: int = 3):
    """
    Search the Chroma vector database.
    """

    query_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    docs = results["documents"][0]

    print("\n========== Chroma Retrieval ==========\n")

    context = ""

    for i, chunk in enumerate(docs, start=1):

        print(f"Chunk {i}")
        print("-" * 40)
        print(chunk[:300])
        print()

        context += f"[Chunk {i}]\n{chunk}\n\n"

    return context


def retrieve_python_docs(question: str):

    print("\n========== Python Docs ==========\n")

    return search_python_docs(question)