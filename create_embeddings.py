import uuid
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF_PATH = "python_tutorial.pdf"
COLLECTION_NAME = "python_tutorial"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ],
)

docs = text_splitter.split_documents(documents)

chunks = [
    doc.page_content
    for doc in docs
]

metadatas = [
    {
        "page": doc.metadata["page"] + 1,   
        "chunk": i
    }
    for i, doc in enumerate(docs)
]

print(f"Created {len(chunks)} chunks")


print("Generating embeddings noe")

embeddings = embedding_model.encode(
    chunks,
    show_progress_bar=True,
    convert_to_numpy=True
).tolist()

db = chromadb.PersistentClient(path="./chroma_db")

try:
    db.delete_collection(COLLECTION_NAME)
    print("Deleted previous collection.")
except Exception:
    print("No previous collection found.")

collection = db.create_collection(COLLECTION_NAME)

collection.add(
    ids=[
        str(uuid.uuid4())
        for _ in chunks
    ],
    documents=chunks,
    embeddings=embeddings,
    metadatas=metadatas,
)

print("\n\nEmbeddings stored successfully!!")
print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print(f"Collection: {COLLECTION_NAME}")
