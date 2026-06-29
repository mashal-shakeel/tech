## Day 3

### 1. Mini RAG System

- Complete RAG pipeline using a PDF document.
- the LLM answers questions using retrieved pdf context instead of relying only on its pretrained knowledge


### 2. PDF Processing

- Used **PyPDF** to load and extract text from a PDF document.


### 3. LangChain Chunking

- Used LangChain for chunking
- Configured chunk overlap to preserve context between chunks
- Split text using multiple separators (`\n\n`, `\n`, sentences, spaces) instead of fixed character slicing

### 4. ChromaDB Vector Store

- Stored document embeddings in ChromaDB
- Embeddings created once and re used

### 5. Semantic Retrieval

- converted the user's question into an embedding
- queried ChromaDB using vector similarity
- retrieved the Top-K (3 in my case) most relevant chunks before sending them to the LLM.


### 6. Grounded Prompting

Built a RAG system prompt that instructs the model to:
  - answer using the retrieved context,
  - combine information across multiple chunks,
  - admit when the answer is missing,
  - avoid hallucinating unsupported facts.
