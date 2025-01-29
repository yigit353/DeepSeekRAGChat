import asyncio
import os
import tempfile
from pathlib import Path

import ollama
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain.chains import (
    create_history_aware_retriever,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from deepseek import DeepseekChatOpenAI

# Load environment variables
load_dotenv()

embedding_model = os.getenv("EMBEDDING_MODEL")
if embedding_model is "openai":
    embeddings = OpenAIEmbeddings()
elif embedding_model is "modernbert":
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-modernbert-base",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={
            "normalize_embeddings": True,
        }
    )
else:
    raise ValueError(f"Embedding model '{embedding_model}' not recognized. "
                     f"Check your environment variables. Set it to 'openai' or 'modernbert'.")

vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)

# Streamlit app for chat and file upload
st.title("LangChain Chat App with Deepseek-R1 and others")
st.write("Upload a document to add to the database and chat with your document database.")

# File upload
uploaded_file = st.file_uploader("Upload your document (.docx, .pdf or .txt):", type=["docx", ".pdf", "txt", ])

if uploaded_file:
    suffix = "".join(Path(uploaded_file.name).suffixes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    if uploaded_file.name.endswith(".docx"):
        doc_content = Docx2txtLoader(tmp_file_path).load()
    elif uploaded_file.name.endswith(".pdf"):
        doc_content = PyPDFLoader(tmp_file_path).load()
    else:
        doc_content = TextLoader(tmp_file_path).load()
    os.unlink(tmp_file_path)  # Clean up temp file

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100
    )
    docs = text_splitter.split_documents(doc_content)
    vectorstore.add_documents(docs)
    st.success(f"File '{uploaded_file.name}' added to the vector database.")


def get_local_ollama_models():
    models = ollama.list()
    return [model["model"] for model in models["models"]]


deepseek_reasoning_models = ["deepseek-reasoner"]
deepseek_models = ["deepseek-chat"]
ollama_models = get_local_ollama_models()
openai_models = ["gpt-4o"]
groq_models = ["deepseek-r1-distill-llama-70b"]

# Model selection dropdown
model_option = st.selectbox(
    "Select Answer Generation Model:",
    options=deepseek_reasoning_models + deepseek_models + ollama_models + openai_models + groq_models,
    index=0
)

# Initialize retriever and LLM
retriever = vectorstore.as_retriever()

if model_option in ollama_models:
    chat_model = OllamaLLM(model=model_option, temperature=0, stream=True)
elif model_option in openai_models:
    chat_model = ChatOpenAI(model_name=model_option)
elif model_option in deepseek_models:
    chat_model = ChatOpenAI(
        model_name=model_option,
        temperature=0,
        streaming=True,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Add this line
        base_url="https://api.deepseek.com/v1"  # Add Deepseek's API endpoint
    )
elif model_option in deepseek_reasoning_models:
    chat_model = DeepseekChatOpenAI(
        model_name=model_option,
        streaming=True,
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )
elif model_option in groq_models:
    chat_model = ChatGroq(model_name=model_option, client=Groq(api_key=os.getenv("GROQ_API_KEY")))
else:
    raise ValueError(f"Model '{model_option}' not recognized.")

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat_model, retriever, contextualize_q_prompt
)

# Define QA prompt
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


class ReasoningCaptureCallback(AsyncCallbackHandler):
    def __init__(self):
        self.answer = ""
        self.reasoning = ""
        self.answer_placeholder = None
        self.reasoning_placeholder = None
        self.reasoning_container = None  # Container for streaming reasoning

    async def on_chat_model_start(self, serialized, messages, **kwargs):
        self.answer_placeholder = st.empty()
        self.reasoning_placeholder = st.expander("View Reasoning Process")
        with self.reasoning_placeholder:
            self.reasoning_container = st.empty()  # Create an empty container

    async def on_llm_new_token(self, token: str, *, chunk: ChatGenerationChunk, **kwargs):
        if chunk.message.additional_kwargs.get("reasoning"):
            reasoning_token = chunk.message.additional_kwargs["reasoning"]
            self.reasoning += reasoning_token
            # Update the same container with accumulated reasoning
            self.reasoning_container.code(self.reasoning)
        elif token:
            self.answer += token
            self.answer_placeholder.markdown(self.answer)


# Helper function to handle async generator
async def process_stream_async(query: str, chat_history: list):
    callback = ReasoningCaptureCallback()

    try:
        # Retrieve context documents using history-aware retriever
        docs = await history_aware_retriever.ainvoke({
            "input": query,
            "chat_history": chat_history
        })

        # Format prompt with retrieved context
        formatted_prompt = await qa_prompt.ainvoke({
            "context": "\n\n".join(doc.page_content for doc in docs),
            "chat_history": chat_history,
            "input": query
        })

        # Stream response directly from the model
        answer = ""
        async for chunk in chat_model.astream(formatted_prompt, config={"callbacks": [callback]}):
            if content := chunk.content:
                answer += content
                if callback.answer_placeholder:
                    callback.answer_placeholder.markdown(answer)

        return answer, docs, callback.reasoning

    except Exception as e:
        st.error(f"Error processing request: {str(e)}")
        return "", [], ""


# Wrapper for asyncio.run
def process_stream_sync(query: str, chat_history: list):
    return asyncio.run(process_stream_async(query, chat_history))


# Chat interface
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["type"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question"):
    with st.chat_message("user"):
        st.write(prompt)
    try:
        with st.spinner("Bot is thinking..."):
            answer, context, reasoning = process_stream_sync(
                prompt,
                st.session_state["chat_history"]
            )

            st.session_state.chat_history.extend([
                {"type": "user", "content": prompt},
                {"type": "assistant", "content": answer, "reasoning": reasoning}
            ])

            with st.chat_message("assistant"):
                st.markdown(answer)
                if reasoning:
                    with st.expander("View Reasoning Process"):
                        st.code(reasoning)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
