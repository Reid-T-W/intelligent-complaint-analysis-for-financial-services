from json import load
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_dataset():
    dataset_path = 'data\cleaned\filtered_dataset.csv'

    loader = CSVLoader(
        file_path=dataset_path,
        content_columns=['Consumer complaint narrative'],
        # source_column='Consumer complaint narrative',
        metadata_columns=['Complaint ID']
    )
    docs = loader.load()
    return docs

def chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts

def embed_and_index(chunked_text):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Convert documents to vectors and index vectors
    db = FAISS.from_documents(chunked_text, embeddings)
    return db

def process():
    docs = load_dataset()
    chunked_text = chunk(docs)
    vector_db = embed_and_index(chunked_text)


if __name__ == "__main__":
    process()