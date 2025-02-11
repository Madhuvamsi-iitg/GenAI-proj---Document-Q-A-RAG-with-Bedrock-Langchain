## Document Q&A RAG with Langchain and Bedrock

This project is a **Document Question-Answering (Q&A) system** built using **LangChain**, **AWS Bedrock**, and **Streamlit**. It leverages the **Retrieval-Augmented Generation (RAG)** approach to answer questions based on the content of PDF documents. Here's a concise summary of the project:

---

### **Project Overview**
1. **Objective**:
   - Build a **Document Q&A system** that allows users to ask questions about the content of PDF documents and get accurate, context-aware answers.

2. **Key Components**:
   - **AWS Bedrock**: Used for generating embeddings (via **Amazon Titan Embeddings**) and running large language models (LLMs) like **Mistral Large** and **Llama3**.
   - **LangChain**: Manages the document processing, embeddings, vector storage, and retrieval-augmented generation.
   - **Streamlit**: Provides a user-friendly web interface for interacting with the system.

3. **Workflow**:
   - **Data Ingestion**: Loads PDF documents from a directory and splits them into smaller chunks for processing.
   - **Embeddings**: Converts document chunks into vector embeddings using **Amazon Titan Embeddings**.
   - **Vector Store**: Stores embeddings in a **FAISS vector database** for efficient similarity search.
   - **Question-Answering**: Retrieves relevant document chunks based on the user's query and generates answers using either **Mistral Large** or **Llama3** LLMs.

4. **Features**:
   - **Dynamic Document Loading**: Users can update the vector store with new PDF documents.
   - **Multiple LLM Support**: Users can choose between **Mistral Large** and **Llama3** for generating answers.
   - **RAG Approach**: Combines retrieval of relevant document chunks with LLM-based answer generation for accurate and context-aware responses.

---

### **Key Functions**
1. **Data Ingestion**:
   - Uses `PyPDFDirectoryLoader` to load PDFs from a directory.
   - Splits documents into chunks using `RecursiveCharacterTextSplitter`.

2. **Vector Store**:
   - Creates and saves embeddings in a **FAISS vector store** for fast retrieval.

3. **LLM Integration**:
   - Supports **Mistral Large** and **Llama3** via AWS Bedrock for generating answers.

4. **Retrieval-Augmented Generation (RAG)**:
   - Retrieves relevant document chunks using FAISS.
   - Uses a custom prompt template to generate