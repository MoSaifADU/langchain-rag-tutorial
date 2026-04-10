import streamlit as st
import os
import shutil
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --- 1. SECRETS & ENVIRONMENT SETUP ---
load_dotenv()

# Define a placeholder for the key
openai_api_key = None

# 1. Try to get key from Streamlit Secrets (Cloud)
try:
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    # If st.secrets doesn't exist (Local), this block will catch the error and move on
    pass

# 2. If Cloud key not found, try to get it from .env or System (Local)
if not openai_api_key:
    openai_api_key = os.getenv("OPENAI_API_KEY")

# 3. Final check and setup
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
else:
    st.error("Missing OpenAI API Key! Please add it to your .env file or Streamlit Secrets.")
    st.stop()


CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
os.makedirs(DATA_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
You are a helpful and detailed assistant. 
Use the following context to provide a comprehensive answer to the user's question.
Use the history to understand follow-up questions.

Context:
{context}

History:
{history}

Question: {question}
"""

st.set_page_config(page_title="Saif's Knowledge Bot", page_icon="🤖")

# --- 2. SIDEBAR (Uploader & Processing) ---
with st.sidebar:
    st.title("📁 Document Management")
    
    uploaded_files = st.file_uploader("Upload PDFs or Markdown", accept_multiple_files=True, type=['pdf', 'md'])
    
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Reading and Indexing documents..."):
                # Save files
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load with error handling
                documents = []
                for file in os.listdir(DATA_PATH):
                    if file.startswith('.'): continue
                    path = os.path.join(DATA_PATH, file)
                    try:
                        if file.endswith(".pdf"):
                            loader = PyPDFLoader(path)
                            documents.extend(loader.load())
                        elif file.endswith(".md"):
                            loader = TextLoader(path, encoding="utf-8")
                            documents.extend(loader.load())
                    except Exception as e:
                        st.warning(f"Skipping {file}: Corrupted or invalid format.")
                
                if not documents:
                    st.error("No readable text found in documents.")
                    st.stop()

                # Split
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)
                
                # Rebuild Vector DB
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                if os.path.exists(CHROMA_PATH):
                    try:
                        # Clear existing data to avoid duplicates/conflicts
                        db_to_clear = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                        db_to_clear.delete_collection()
                    except:
                        pass
                
                Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
                st.success(f"Success! {len(chunks)} chunks indexed.")
        else:
            st.warning("Upload files first!")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 3. MAIN CHAT INTERFACE ---
st.title("🤖 Saif's Knowledge Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # We retrieve more chunks (k=10) to ensure we find the answer
    results = db.similarity_search_with_relevance_scores(prompt, k=10)
    
    # Check if we actually found anything
    if not results or len(results) == 0:
        response_text = "I couldn't find any information in the documents. Did you click 'Process Documents'?"
    else:
        # Sort and filter results (optional: removes very low relevance matches)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc, _score in results]))
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])
        
        final_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, history=history_text, question=prompt
        )
        
        model = ChatOpenAI(model="gpt-4o-mini")
        response_text = model.invoke(final_prompt).content
        
        source_names = [os.path.basename(s) for s in sources]
        response_text += f"\n\n**Sources:** {', '.join(source_names)}"

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
