import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("NewsLens: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "news-lens-chatbot"

pc = Pinecone(api_key=pinecone_api_key)

# Use session state to maintain vectorstore across reruns
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # Delete existing index if it exists
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name in existing_indexes:
        pc.delete_index(index_name)
        main_placeholder.text("Deleted existing index...✅✅✅")
        time.sleep(1)

    # Create new index
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name)["status"]["ready"]:
        time.sleep(1)
    main_placeholder.text("Created new index...✅✅✅")
    time.sleep(1)

    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save them to Pinecone index
    embeddings = OpenAIEmbeddings()
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    
    st.session_state.vectorstore = PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index_name=index_name
    )
    main_placeholder.text("Vectors Uploaded to Pinecone...✅✅✅")
    time.sleep(2)

query = main_placeholder.text_input("Question: ")

def retrieve_query(query, k=4):
    if st.session_state.vectorstore is None:
        st.warning("Please process URLs first by clicking the 'Process URLs' button.")
        return None
    matching_results = st.session_state.vectorstore.similarity_search(query, k=k)
    return matching_results

if query:
    if st.session_state.vectorstore is not None:
        doc_search = retrieve_query(query)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever()
        )
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
    else:
        st.warning("Please process URLs first by clicking the 'Process URLs' button.")