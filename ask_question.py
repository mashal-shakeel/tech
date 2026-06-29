import os
import chromadb
from huggingface_hub.errors import BadRequestError
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(api_key=HF_TOKEN)

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_collection("python_tutorial")

question = input("\nAsk a question: ")

query_embedding = embedding_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]

print("\nRetrieved Context\n")

context = ""

for i, chunk in enumerate(retrieved_chunks, start=1):

    print(f"Chunk {i}")
    print("-" * 40)
    print(chunk[:300], "...\n")

    context += f"[Chunk {i}]\n{chunk}\n\n"

system_prompt = """
                You are a Retrieval-Augmented Generation (RAG) assistant.
                You will receive context retrieved from a document.

                Instructions:
                - Answer using the retrieved context whenever possible.
                - You may combine information from multiple retrieved chunks.
                - If the context partially answers the question, provide the partial answer and mention what information is missing.
                - Do not ignore relevant information found in later chunks.
                - If the answer is completely absent, reply: "I don't know based on the provided document."
                - Do not invent facts that are not supported by the context.
                - Keep your answer concise and clear.
                """

user_prompt = f"""
            Context:

            {context}

            -----------------------

            Question:

            {question}
            """

try:
    print("\nPrompt length:", len(user_prompt), "characters")
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    print("\n\n\nFINAL ANSWER!!\n")

    print(response.choices[0].message.content)

except BadRequestError as e:

    print("\n FULL ERROR \n")

    print(e)

    if hasattr(e, "response") and e.response is not None:
        print(e.response.text)

    raise