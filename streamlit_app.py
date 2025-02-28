import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Streamlit app introduction
st.title("FDA Cosmetic Guidance Q&A Chatbot")
st.write("Ask questions about FDA cosmetic regulations.")

# Sidebar settings
st.sidebar.header("Settings")
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
chunk_size = st.sidebar.slider("Document Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Document Chunk Overlap", 50, 500, 100, 50)

# Process the FDA Cosmetic Guidance document
@st.cache_data
def load_documents(file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = load_documents("FDA_guidance.pdf", chunk_size, chunk_overlap)

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = FAISS.from_documents(docs, embeddings)

# Initialize the LLM
llm = OpenAI(temperature=temperature, openai_api_key=st.secrets["OPENAI_API_KEY"])

# Define a custom prompt template for RAG
prompt_template = """
Use the following context to answer the question. Provide citations from the context where applicable.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Add chat memory for context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    memory=memory
)

# Display chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
query = st.text_input("Enter your regulatory query:")

if query:
    with st.spinner("Processing your query..."):
        # Run the RAG chain
        result = qa_chain({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])

        # Update chat history
        st.session_state.chat_history.append({"question": query, "answer": answer, "sources": sources})

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)

    # Display sources (citations)
    if sources:
        st.subheader("Sources:")
        for i, source in enumerate(sources):
            st.write(f"**Source {i+1}:**")
            st.write(source.page_content)  # Display the relevant document chunk
            st.write("---")

# Display chat history
st.sidebar.header("Chat History")
for chat in st.session_state.chat_history:
    st.sidebar.write(f"**Q:** {chat['question']}")
    st.sidebar.write(f"**A:** {chat['answer']}")
    st.sidebar.write("---")

# Clear chat history
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.write("Chat history cleared.")
