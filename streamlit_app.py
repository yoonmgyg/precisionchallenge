import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain


st.title("FDA Cosmetic Guidance Q&A Chatbot")
st.write("Ask questions about FDA cosmetic regulations and receive responses with citations.")


st.sidebar.header("Settings")
temperature = st.sidebar.slider("LLM Temperature", 0.0, 1.0, 0.0, 0.1)
chunk_size = st.sidebar.slider("Document Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Document Chunk Overlap", 50, 500, 100, 50)

@st.cache_data
def load_documents(file_path, chunk_size, chunk_overlap):
    # Load the document from a PDF file
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Split the text into manageable chunks
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

docs = load_documents("FDA_guidance.pdf", chunk_size, chunk_overlap)

# Make sure your OpenAI API key is stored in Streamlit secrets (st.secrets["OPENAI_API_KEY"])
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
vectorstore = FAISS.from_documents(docs, embeddings)


llm = OpenAI(temperature=temperature, openai_api_key=st.secrets["OPENAI_API_KEY"])


qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


query = st.text_input("Enter your regulatory query:")

if query:
    with st.spinner("Processing your query..."):
        result = qa_chain({"question": query})
    st.subheader("Answer:")
    st.write(result["answer"])
    

