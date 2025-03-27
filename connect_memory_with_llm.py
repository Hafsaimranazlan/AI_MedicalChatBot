from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS
load_dotenv()
# setup LLM

hf_token=os.environ.get("HF_TOKEN")

model_repo_id="mistralai/Mistral-7B-Instruct-v0.3"
print("model loading.....")
llm=HuggingFaceEndpoint(
    repo_id=model_repo_id,
    model_kwargs={
        'token':hf_token,
        'max_length':'512'
    }
)
print("model loaded")

# connect LLM with faiss and create chain

DB_FAISS_PATH="vectorstore_df_faiss"

custom_prompts_template=""""
use the pieces of information provide in the context to answer user's question.
if you donot the answer, just say that you donot know, donot try to make up an answer,
Donot provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""
prompt=PromptTemplate(
    template=custom_prompts_template,
    input_variables=['context','question']
)


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db=FAISS.load_local(DB_FAISS_PATH,embed_model,allow_dangerous_deserialization=True)




qa_chain=RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':prompt}

)



while True:
    # Prompt the user for input
    user_query = input("Write Query Here (or type 'quit' to exit): ")

    # Check if the user wants to quit
    if user_query.lower() in ["quit", "exit"]:
        print("Exiting the program. Goodbye!")
        break

    # Process the query
    response = qa_chain.invoke({'query': user_query})

    # Print the result and source documents
    print("Response:")
    print(response["result"])
    print("\n Response Documents:")
    print(response['source_documents'])