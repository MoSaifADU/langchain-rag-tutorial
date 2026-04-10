import streamlit as st
import os
import shutil
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    
# Setup
load_dotenv()
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"
os.makedirs(DATA_PATH, exist_ok=True)

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the question based ONLY on the following context. 
If the answer isn't in the context, say you don't know. 

Context:
{context}

History:
{history}

Question: {question}
"""

st.set_page_config(page_title="Saif's RAG Bot", page_icon="🚀")

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings & Upload")
    
    # 1. File Uploader
    uploaded_files = st.file_uploader("Upload new documents", accept_multiple_files=True, type=['pdf', 'md'])
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                # Save uploaded files to the data folder
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load all documents from the folder
                documents = []
                for file in os.listdir(DATA_PATH):
                    path = os.path.join(DATA_PATH, file)
                    if file.endswith(".pdf"): documents.extend(PyPDFLoader(path).load())
                    elif file.endswith(".md"): documents.extend(TextLoader(path, encoding="utf-8").load())
                
                # Split into chunks
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)
                
                # WINDOWS FIX: Clear existing collection instead of deleting the folder
                # This avoids the "File in Use" PermissionError
                embeddings = OpenAIEmbeddings()
                if os.path.exists(CHROMA_PATH):
                    db_to_clear = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
                    db_to_clear.delete_collection() 
                
                # Rebuild database
                Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
                st.success(f"Database Updated! {len(chunks)} chunks saved.")
        else:
            st.warning("Please upload files first.")

    # 2. Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🤖 Saif's Knowledge Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about your files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG Logic
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(prompt, k=8)
    
    # Use a lower threshold for better retrieval on mixed documents
    if len(results) == 0 or results[0][1] < 0.3:
        response_text = "I couldn't find relevant info in the docs."
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = list(set([doc.metadata.get("source", "Unknown") for doc, _score in results]))
        
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:-1]])
        
        final_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
            context=context_text, history=history_text, question=prompt
        )
        
        model = ChatOpenAI(model="gpt-4o-mini")
        response_text = model.invoke(final_prompt).content
        # Append source filenames for transparency
        source_names = [os.path.basename(s) for s in sources]
        response_text += f"\n\n**Sources:** {', '.join(source_names)}"

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
