"""
Wikipedia RAG with Chroma & FAISS

This project implements a Retrieval-Augmented Generation (RAG) system using data 
extracted from the Simple English Wikipedia. It leverages LangChain to orchestrate the process,
Sentence Transformers for creating embeddings, uses ChromaDB and FAISS as vector stores for retrieval,
and uses a LLM via LangChain for generating answers based on the retrieved context.

Link to the GitHub repository: https://github.com/jean-ferrer/WikiRAG
"""

### Imports ###

from pathlib import Path
from tqdm import tqdm
import shutil
import json
import re
import os

from API_TOKEN import API_TOKEN
from huggingface_hub import snapshot_download

from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
import pickle

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold




### Load the Documents ###

wiki_docs = []

docs_path = Path("docs")

# Searches for all files in subfolders, regardless of the extension
for file_path in docs_path.rglob("*"):
    if file_path.is_file(): # ignores folders
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    article = json.loads(line)
                    text = article.get("text", "").replace("\n", " ").strip()

                    if text:
                        wiki_docs.append(
                            Document(
                                page_content=text,
                                metadata={
                                    "title": article.get("title", "Unknown"),
                                    "source_file": str(file_path.relative_to(docs_path)) # example: "AA/wiki_00"
                                }
                            )
                        )
        except Exception as e:
            print(f"Error processing {file_path}: {e}")




### Create Embeddings ###

# Code to download all-MiniLM-L6-v2 #

# Destination path within your project
local_model_path = "sentence-transformers/all-MiniLM-L6-v2"

# Checks if the folder already exists and contains files
if os.path.exists(local_model_path) and os.listdir(local_model_path):
    print(f"Model already exists at: {local_model_path}")
else:
    # Downloads the model to the local Hugging Face cache
    cached_model_path = snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2")

    # Creates the destination directory if it doesn't exist
    os.makedirs(local_model_path, exist_ok=True)

    # Copies all files from the cache to the project folder
    for item in os.listdir(cached_model_path):
        s = os.path.join(cached_model_path, item)
        d = os.path.join(local_model_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print(f"Model copied to: {local_model_path}")


# Embedding function using a light and efficient model
embeddings = SentenceTransformerEmbeddings(model_name=local_model_path)




### Implement Chroma Vector Database ###

# Define the Chroma directory
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "SimpleWiki"

# Check if the directory already exists to avoid recreating the vector store
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("🔄 Loading existing vector store...")
    vector_db_chroma = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
else:
    print("🆕 Creating new Chroma vector store and saving to disk...")

    # Initialize Chroma with embedding function and persist directory
    vector_db_chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )

    # Separate texts and metadata
    texts = [doc.page_content for doc in wiki_docs]
    metadatas = [doc.metadata for doc in wiki_docs]

    # Generate embeddings for Chroma
    print("📊 Generating embeddings for Chroma...")
    embeddings_list_chroma = [embeddings.embed_query(text) for text in tqdm(texts, desc="📥 Embedding (Chroma)")]
    print(f"✅ Embeddings generated for {len(embeddings_list_chroma)} documents (Chroma).")

    # Create documents in Langchain format
    documents = [Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))]

    # Add documents with their embeddings to the vector store
    print("📥 Adding documents to the Chroma vector store...")
    for i in tqdm(range(len(documents)), desc="➕ Adding to Chroma"):
        vector_db_chroma.add_documents(documents=[documents[i]], embeddings=[embeddings_list_chroma[i]])

# Convert the vector store to a retriever with MMR search
chroma_retriever = vector_db_chroma.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

print("\n✅ Chroma vector database and retriever are ready.")
print(f"Retriever type: {chroma_retriever.__class__.__name__}")


# Function to extract semantically relevant text with Chroma
def extract_text_chroma(query: str, max_words_per_doc: int = 200) -> str:
    """
    Retrieves snippets of text semantically relevant to the query using a Retriever,
    with a word limit per document and removal of exact duplicates.
    """
    results = chroma_retriever.invoke(query)
    if not results:
        return "No matching information found."

    seen = set()
    cleaned_chunks = []

    for doc in results:
        content = doc.page_content.strip()
        words = re.findall(r'\S+', content)
        trimmed = " ".join(words[:max_words_per_doc])
        normalized = re.sub(r"\s+", " ", trimmed).strip()

        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            cleaned_chunks.append(trimmed)

    return "\n\n".join(cleaned_chunks)

query = 'What is the average height of a peach tree?'
print(extract_text_chroma(query))




### Implement FAISS Vector Database ###

# Define the FAISS directory and filename
FAISS_DIR = "faiss_db"
FAISS_INDEX_FILE = "faiss_index.pkl"
FAISS_PATH = Path(FAISS_DIR) / FAISS_INDEX_FILE

# Initialize FAISS index
embedding_dimension = len(embeddings.embed_query("hello world"))

# Check if the directory already exists to avoid recreating the vector store
if os.path.exists(FAISS_PATH):
    print("🔄 Loading existing FAISS vector store...")
    # Load the entire vector_db_faiss object (which includes index, docstore, etc.)
    with open(FAISS_PATH, "rb") as f:
        vector_db_faiss = pickle.load(f)
else:
    print("🆕 Creating new FAISS vector store and saving to disk...")

    # Create FAISS directly from documents
    print("📊 Generating embeddings and building FAISS from documents...")
    vector_db_faiss = FAISS.from_documents(wiki_docs, embeddings) # This step takes a while, be patient
    print(f"✅ FAISS vector store created and populated with {len(wiki_docs)} documents.")

    # Save FAISS vector store to disk
    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FAISS_PATH, "wb") as f:
        pickle.dump(vector_db_faiss, f) # Pickle the entire vector_db_faiss object
    print(f"💾 FAISS vector store saved to: {FAISS_PATH}")

# Convert the FAISS vector store to a retriever
faiss_retriever = vector_db_faiss.as_retriever(search_kwargs={"k": 3})

print("\n✅ FAISS vector database and retriever are ready.")
print(f"Retriever type: {faiss_retriever.__class__.__name__}")


# Function to extract semantically relevant text with FAISS
def extract_text_faiss(query: str, retriever=faiss_retriever, max_words_per_doc: int = 200) -> str:
    """
    Retrieves snippets of text semantically relevant to the query using a Retriever,
    with a word limit per document and removal of exact duplicates.
    """
    # Use the retriever's invoke method to get the relevant documents
    results: list[Document] = retriever.invoke(query)

    if not results:
        return "No matching information found."

    seen = set()
    cleaned_chunks = []

    for doc in results:
        # Ensure the document content is a string
        content = str(doc.page_content).strip()

        # Split content into words and take the first max_words_per_doc
        words = re.findall(r'\S+', content)
        trimmed = " ".join(words[:max_words_per_doc])

        # Normalize whitespace and convert to lowercase for duplicate checking
        normalized = re.sub(r"\s+", " ", trimmed).strip().lower()

        # Add the trimmed chunk if its normalized version hasn't been seen before
        if normalized not in seen:
            seen.add(normalized)
            # Append the original trimmed version (not lowercase)
            cleaned_chunks.append(trimmed)

    if not cleaned_chunks:
         return "No relevant information found after processing."

    return "\n\n".join(cleaned_chunks)

query = 'What is the average height of a peach tree?'
print(extract_text_faiss(query))




### LLM Setup ###

# Checking Google's available LLM models

print("Listing available models...")
genai.configure(api_key=API_TOKEN)
for model in genai.list_models():
  # Checks if the model supports the generateContent operation (for chat/text)
  if 'generateContent' in model.supported_generation_methods:
    print(f"- Name: {model.name}")
    print(f"  Supported Methods: {model.supported_generation_methods}")
    print("-" * 20)

print("Complete listing.")


# Define safety settings ("guard rails")
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


# Initializes the Google Gemini Flash language model for chat
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_TOKEN, temperature=0.1) # Sets a low temperature for more direct and deterministic responses




### RAG Setup ###

# This template instructs the LLM on how to use the context.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and informative assistant.
      Use ONLY the following provided context to answer the question.
      If you cannot find the answer in the provided context, simply say that you do not have enough information.
      Do not use your prior knowledge.
      Ensure the answer is clear and concise."""),
    ("user", "Context: {context}\n\nQuestion: {input}")
])

# First, we create a chain that "stuffs" all the retrieved documents into the {context} slot of the prompt.
document_chain = create_stuff_documents_chain(llm, prompt)

# Defining the two retrievers
rag_chain_chroma = create_retrieval_chain(chroma_retriever, document_chain)
rag_chain_faiss = create_retrieval_chain(faiss_retriever, document_chain)

# Making the question
query_rag = "What is the average height of a peach tree?"




### Chroma RAG ###

# Print the question being asked
print(f"\n❓ Asking: {query_rag}")

# The result will contain the answer ('answer') and the source documents ('context') retrieved from Chroma
response = rag_chain_chroma.invoke({"input": query_rag})

# Print the LLM's response based on the retrieved context
print("\n--- LLM Response ---")
print(response["answer"])


# Print the source documents that were used
print("\n--- Source Documents (Context) ---")
for doc in response["context"]:
    # Print document metadata (title and source file) if available
    print(f"- {doc.metadata.get('title', 'Untitled')} (Source: {doc.metadata.get('source_file', 'N/A')})")
    # Print the first 500 characters of the document content
    print(doc.page_content[:500] + "...")




### FAISS RAG ###

# Print the question being asked
print(f"\n❓ Asking: {query_rag}")

# The result will contain the answer ('answer') and the source documents ('context') retrieved from FAISS
response = rag_chain_faiss.invoke({"input": query_rag})

# Print the LLM's response based on the retrieved context
print("\n--- LLM Response ---")
print(response["answer"])


# Print the source documents that were used
print("\n--- Source Documents (Context) ---")
for doc in response["context"]:
    # Print document metadata (title and source file) if available
    print(f"- {doc.metadata.get('title', 'Untitled')} (Source: {doc.metadata.get('source_file', 'N/A')})")
    # Print the first 500 characters of the document content
    print(doc.page_content[:500] + "...")