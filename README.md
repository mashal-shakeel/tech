## Day 4

### 1. ReAct Tool-Calling Agent

- Built a simple ReAct-based AI agent
- agent reasons step-by-step before answeringg
- It can call external tools when additional information is needed

### 2. Tool Integration

Integrated tools:

- **`retrieve_documents()`** – retrieves relevant chunks from the ChromaDB knowledge base built in Day 3.
- **`search_python_docs()`** – searches and extracts content from the official Python documentation on web

### 3. Agent Loop

Implemented a reasoning loop that:

- generates a thought
- executes one tool at a time
- observes the tool output
- repeats until a Final Answer is produced or the maximum step limit is reached

### 4. Human Approval Gate

- Added a confirmation step before executing every tool call.
- Tool execution proceeds only after user approval.

### 5. Agent Prompting
system prompt instructs model to reason before acting then choose the correct tool and call only one tool at a time while avoiding inventing tool outputs. Finally it stops only after enough information has been gathered

### 6. Built on Day 3

Reused the Mini RAG pipeline from Day 3:

- ChromaDB vector store
- SentenceTransformer embeddings
- LangChain chunking
- Semantic document retrieval