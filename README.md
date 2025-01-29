# LangChain Chat App with Deepseek-R1 and Others

Welcome to this simple demonstration of how to integrate various LLMs and embeddings into a Streamlit-powered application. This project shows how you can upload and parse documents, store them in a vector database, and query them with different language models.

## Key Points
- **Main Goal**: Retrieve reasoning tokens from the DeepSeek API, which come through a separate channel from the primary content, and display them in a nice UI while streaming.
- **Streaming Fix**: Solve the streaming issue with DeepSeek API by covering ChatOpenAI for LangChain integration, which is not yet standardized.
- **Multi-turn Retrieval**: Support multi-turn conversation with retrieval logic.
- **File-based Persistence**: Enable uploads of multiple data types (PDF, DOCX, TXT) and persist embeddings in the file system.

## Overview

This project focuses on a conversational interface that:
- Allows users to upload `.docx`, `.pdf`, or `.txt` files.
- Splits the document content into manageable chunks and stores them in a [Chroma](https://github.com/chroma-core/chroma) vectorstore.
- Provides an interactive chat interface where users can ask questions that leverage the stored content.
- Integrates with various models, including:
  - **Deepseek** (both Chat and Reasoning flavors)
  - **Ollama** for local inference
  - **OpenAI** (e.g., GPT-4 variants)
  - **Groq** for specialized deployments

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yigit353/DeepSeekRAGChat.git
   cd DeepSeekRAGChat
   ```

2. **Create and Activate a Virtual Environment (Optional but Recommended)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in your project root. It should contain the environment variables needed:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   GROQ_API_KEY=your_groq_api_key
   EMBEDDING_MODEL=openai  # or 'modernbert'
   ```

   - Adjust these environment variables based on your desired configuration.

5. **Chroma Persistence Directory**:
   - By default, the vector database is persisted at `./db`. If you wish to change this location, edit the code accordingly.

## Usage

1. **Start the Streamlit App**:
   ```bash
   streamlit run chat.py
   ```

2. **Open the App**:
   - By default, Streamlit apps open at `http://localhost:8501`.

3. **Upload Documents**:
   - In the web interface, upload `.docx`, `.pdf`, or `.txt` files.
   - The application extracts text using the appropriate loader, splits it into chunks, and inserts those chunks into the Chroma vectorstore.

4. **Chat With Your Data**:
   - Choose which model you want to use for question answering.
   - Enter your queries in the chat input box.
   - The app retrieves the most relevant documents to provide context.
   - The chosen model generates an answer based on that context.

5. **View Reasoning** (Optional):
   - When the model responds, an expander titled **"View Reasoning Process"** will appear.
   - Click the expander to see the step-by-step reasoning tokens as they stream from the model.


## Supported Models

### 1. Deepseek
- **DeepseekChatOpenAI**: A specialized wrapper for Deepseek's chat endpoint.
- **DeepseekChatOpenAI (reasoner)**: Similar to chat but optimized for more advanced reasoning.

### 2. Ollama
- **OllamaLLM**: Uses your local [Ollama](https://github.com/jmorganca/ollama) models.

### 3. OpenAI
- **ChatOpenAI**: GPT-4o model for conversational chat.

### 4. Groq
- **ChatGroq**: Specialized client with Groq. 70b DeepSeek Reasoner model with super fast generation.

## Customization

- **Chunk Size / Overlap**: Modify the `RecursiveCharacterTextSplitter` parameters to suit your document size and retrieval needs.
- **Retrieval**: Explore [LangChain's retrieval features](https://python.langchain.com/en/latest/modules/indexes/retrievers.html) to implement advanced logic.
- **Model Configuration**: Adjust temperature, max tokens, etc., when instantiating your models.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

MIT License. See [LICENSE](LICENSE) for more information.

---

### Contact

If you have any questions or need further help:

- **Email**: [yigit353@gmail.com](mailto:yigit353@gmail.com)
- **GitHub Issues**: [Issues Page](https://github.com/yigit353/DeepSeekRAGChat/issues)

Enjoy chatting with your documents!

