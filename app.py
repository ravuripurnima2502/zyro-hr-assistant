import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
page_title="Zyro Dynamics HR Assistant",
page_icon="💼",
layout="wide"
)

st.title("💼 Zyro Dynamics HR Assistant")
st.caption("Ask questions about company HR policies")

@st.cache_resource
def load_rag():

```
loader = PyPDFDirectoryLoader("hr_corpus")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20
    }
)

return retriever
```

retriever = load_rag()

llm = ChatGroq(
model="llama-3.3-70b-versatile",
temperature=0,
api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
"""
You are Zyro Dynamics HR Assistant.

Answer ONLY using the provided HR policy context.

If the answer is not available in the context, reply exactly:

I can only answer HR-related questions based on Zyro Dynamics policy documents.

Context:
{context}

Question:
{question}

Answer:
"""
)

chain = (
prompt
| llm
| StrOutputParser()
)

if "messages" not in st.session_state:
st.session_state.messages = []

for message in st.session_state.messages:
with st.chat_message(message["role"]):
st.write(message["content"])

query = st.chat_input("Ask an HR question...")

if query:

```
st.session_state.messages.append(
    {
        "role": "user",
        "content": query
    }
)

with st.chat_message("user"):
    st.write(query)

docs = retriever.invoke(query)

context = "\n\n".join(
    doc.page_content for doc in docs
)

answer = chain.invoke(
    {
        "context": context,
        "question": query
    }
)

with st.chat_message("assistant"):

    st.write(answer)

    with st.expander("📄 Source Documents"):

        sources = set()

        for doc in docs:
            sources.add(
                os.path.basename(
                    doc.metadata.get(
                        "source",
                        "Unknown"
                    )
                )
            )

        for source in sorted(sources):
            st.write(source)

st.session_state.messages.append(
    {
        "role": "assistant",
        "content": answer
    }
)
```
