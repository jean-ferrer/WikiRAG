# Wikipedia RAG with Chroma & FAISS

This project implements a Retrieval-Augmented Generation (RAG) system using data extracted from the Simple English Wikipedia. It leverages LangChain to orchestrate the process, Sentence Transformers for creating embeddings, uses ChromaDB and FAISS as vector stores for retrieval, and uses a LLM via LangChain for generating answers based on the retrieved context.

## Features

* Processes Simple English Wikipedia XML dumps using WikiExtractor.
* Loads and cleans extracted text data.
* Generates text embeddings using the `all-MiniLM-L6-v2` model (downloaded locally).
* Builds and persists vector stores using both ChromaDB and FAISS.
* Implements RAG pipelines for both Chroma and FAISS retrievers.
* Uses a LLM (Google Gemini) via LangChain for answer generation.
* Defines safety settings (guardrails) for content moderation with the LLM.
* Compares the RAG output using context retrieved from both vector stores.
* Includes helper functions for testing retrieval directly from vector stores.

## Setup Instructions

1.  **Windows Setup (Skip if using Linux/macOS)**
    * If you don't have WSL installed, open PowerShell as Administrator and run:

        ```powershell
        wsl --install
        ```
    * Restart your computer after installation.
    * Verify WSL is working by listing installed distributions:

        ```powershell
        wsl --list --verbose
        ```
    * You can proceed with the next steps using the WSL terminal directly or via a compatible editor like VS Code (use the "Reopen in WSL" option).
  
2.  **Clone This Repository**
    Clone this project repository to your local machine.
    ```bash
    git clone https://github.com/jean-ferrer/WikiRAG
    cd https://github.com/jean-ferrer/WikiRAG
    ```

3.  **Download Wikipedia Data**
    * Download the Simple English Wikipedia dump file `simplewiki-latest-pages-articles.xml.bz2` from: [https://dumps.wikimedia.org/simplewiki/latest/](https://dumps.wikimedia.org/simplewiki/latest/)
    * Place the downloaded `.bz2` file directly into the main project folder (the one you just cloned).

4.  **Extract Wikipedia Text using WikiExtractor**
    * Clone the (fixed) WikiExtractor tool into the main project folder:
        ```bash
        git clone https://github.com/jean-ferrer/wikiextractor.git
        ```
    * Navigate into the `wikiextractor` folder:
        ```bash
        cd wikiextractor
        ```
        *(Example full path might be `/mnt/c/Users/username/Desktop/Project/wikiextractor` if using WSL from Windows Desktop)*  

    * Run the WikiExtractor script. This command assumes the `.bz2` file is in the *parent* folder (`../`) and outputs the extracted text into a new `docs` folder in the *parent* project folder (`../docs`).
        ```bash
        python3 -m wikiextractor.WikiExtractor ../simplewiki-latest-pages-articles.xml.bz2 --output ../docs --json
        ```
    * Navigate back to the main project folder:
        ```bash
        cd ..
        ```

5.  **Set up Python Environment and Install Dependencies**
    * Create a virtual environment (recommended):

        ```bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows WSL/Linux/macOS
        # or venv\Scripts\activate   # On Windows Command Prompt/PowerShell
        ```
    * Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```

6.  **Configure Google API Key**
    * Create a Google API key: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
    * Inside `API_TOKEN.py`, replace `"YOUR_API_KEY"` with your actual Google AI Studio API key:

        ```python
        API_TOKEN = "YOUR_API_KEY"
        ```
    * *Note:* Ensure this file is added to your `.gitignore` if you plan to push your code to a public repository to avoid exposing your key.

## How the Code Works

1.  **Imports:** Imports necessary libraries from LangChain, Sentence Transformers, Hugging Face Hub, Google Generative AI, Chroma, FAISS, and standard Python modules. It also imports your API key from the `API_TOKEN.py` file.  

2.  **Load Documents:**
    * Searches the `docs/` directory (created by WikiExtractor) for extracted article files.
    * Reads each file line by line (each line is a JSON object representing an article).
    * Extracts the `text` and `title` from each article.
    * Creates LangChain `Document` objects, storing the text content and metadata (title, source file path).

3.  **Create Embeddings:**
    * Checks if the `sentence-transformers/all-MiniLM-L6-v2` model exists locally in the specified path (`sentence-transformers/all-MiniLM-L6-v2`).
    * If not found, it downloads the model from Hugging Face Hub using `snapshot_download` and copies it to the local path.
    * Initializes `SentenceTransformerEmbeddings` using the local model path.

4.  **Implement Chroma Vector Database:**
    * Checks if a Chroma database exists at `./chroma_db`.
    * If it exists, it loads the existing database.
    * If not, it creates a new Chroma database:
        * Generates embeddings for all loaded documents.
        * Adds the documents and their embeddings to the Chroma store.
        * Persists the database to the `./chroma_db` directory.
    * Creates a Chroma retriever using `as_retriever` with MMR (Maximal Marginal Relevance) search.
    * Includes a function `extract_text_chroma` to test retrieving relevant text snippets directly.

5.  **Implement FAISS Vector Database:**
    * Checks if a FAISS index file exists at `./faiss_db/faiss_index.pkl`.
    * If it exists, it loads the existing database using `pickle`.
    * If not, it creates a new FAISS database:
        * Uses `FAISS.from_documents` to directly build the index from the documents and embeddings (this can take time).
        * Saves the FAISS database object to `./faiss_db/faiss_index.pkl` using `pickle`.
    * Creates a FAISS retriever using `as_retriever`.
    * Includes a function `extract_text_faiss` to test retrieving relevant text snippets directly.

6.  **LLM Setup:**
    * Configures the Google Generative AI library with the API key.
    * (Optional) Lists available Google models supporting content generation.
    * Initializes the `ChatGoogleGenerativeAI` model (`gemini-2.0-flash`) with a low temperature for more deterministic responses.

7.  **RAG Setup:**
    * Defines a `ChatPromptTemplate` instructing the LLM on how to use the provided context to answer questions.
    * Creates a `document_chain` using `create_stuff_documents_chain` to combine retrieved documents into the prompt's context.
    * Creates two separate RAG chains (`rag_chain_chroma`, `rag_chain_faiss`) using `create_retrieval_chain`, linking the respective retriever (Chroma or FAISS) with the document chain.

8.  **Run RAG Queries:**
    * Defines a sample query (`query_rag`).
    * Invokes the Chroma RAG chain with the query, prints the LLM's answer and the source documents used.
    * Invokes the FAISS RAG chain with the query, prints the LLM's answer and the source documents used.

## Dependencies

Key libraries used:

* `langchain`: Core framework for building LLM applications.
* `langchain-google-genai`: Integration with Google Gemini models.
* `langchain-chroma`: Integration with ChromaDB vector store.
* `langchain-community`: Community integrations, including FAISS.
* `sentence-transformers`: For generating text embeddings.
* `huggingface_hub`: For downloading models from the Hugging Face Hub.
* `google-generativeai`: Google's Python SDK for generative models.
* `tqdm`: Progress bars for long-running operations.

Install all dependencies using `pip install -r requirements.txt`.

## Configuration

* **API Key:** Requires a Google AI Studio API key configured in `API_TOKEN.py`.
* **Embedding Model:** Uses `sentence-transformers/all-MiniLM-L6-v2`, downloaded to `./sentence-transformers/all-MiniLM-L6-v2/`.
* **Vector Stores:**
    * ChromaDB is persisted in the `./chroma_db/` directory.
    * FAISS index is persisted in `./faiss_db/faiss_index.pkl`.
* **LLM:** Uses `gemini-2.0-flash` by default.
