from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings




# load pdf

data_path="data/"

loader=DirectoryLoader(
    data_path,
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

document=loader.load()


# create chunks

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

text_chunk=text_splitter.split_documents(document)

print(len(text_chunk))


# create vector embeddings


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Access the underlying SentenceTransformer object
sentence_transformer = embed_model._client

# Specify the directory to save the model
save_directory = "local_models/all-MiniLM-L6-v2"

# Save the model to the local directory
sentence_transformer.save(save_directory)

print(f"Model saved to {save_directory}")


# store vector embeddings in index

DB_FAISS_PATH="vectorstore_df_faiss"

db=FAISS.from_documents(text_chunk,embed_model)

db.save_local(DB_FAISS_PATH)

