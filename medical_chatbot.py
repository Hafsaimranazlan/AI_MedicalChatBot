import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import os
load_dotenv()


DB_FAISS_PATH="vectorstore_df_faiss"
@st.cache_resource
def get_vectorstore():
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db=FAISS.load_local(DB_FAISS_PATH,embed_model,allow_dangerous_deserialization=True)
    return db

hf_token=os.environ.get("HF_TOKEN")

def main():
    st.title("Ask ChatBot")

    if 'messages' not in st.session_state:
        st.session_state.messages=[]

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    memory = st.session_state.memory 

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['context'])

    
    prompt=st.chat_input("pass your prompt here ...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','context':prompt})


        system_template="""
        use the pieces of information provide in the context to answer user's question.
        if you donot the answer, just say that you donot know, donot try to make up an answer,
        Donot provide anything out of the given context

        Context: {context}

        Start the answer directly. No small talk please.
        """

        #prompts_template=PromptTemplate(
        #    template=custom_prompts_template,
         #   input_variables=['context','question']
        #)

        user_template = "Question: {question}"

        # Create the ChatPromptTemplate
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template),
        ]
        chat_prompt = ChatPromptTemplate.from_messages(messages)

        #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,output_key="answer")

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
        try:
            db=get_vectorstore()
            if db is None:
                st.error("Failed to load the vecotr store")
           #qa_chain=RetrievalQA.from_chain_type(
               # llm=llm,
              #  chain_type='stuff',
             #   retriever=db.as_retriever(search_kwargs={'k':3}),
            #  return_source_documents=True,
             #  chain_type_kwargs={'prompt':prompts_template}

            #)
            print('qa_chain started')
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={'k': 3}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": chat_prompt},      
                #return_source_documents=True,
                output_key="answer"
            )

            print("qa_chain ended")

            response = qa_chain.invoke({'question': prompt })
            result=response["answer"]

            #source_document=response['source_documents']

            #result_show=result+str(source_document)
            #result_show = result + "\n\nSource Documents:\n" + "\n".join([doc.page_content for doc in source_document])
            
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant','context':result})
        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()