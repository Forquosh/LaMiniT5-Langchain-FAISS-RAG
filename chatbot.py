import streamlit as st
from streamlit_chat import message
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage
    
# TODO implement working chat history

# Load model ->
checkpoint = "LaMini-T5-738M"

# Initialize chat history
chat_history = []

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint, 
    device_map='auto', 
    torch_dtype=torch.float32,
    trust_remote_code=True
)

@st.cache_resource
def llm_pipeline():

    # Create pipeline ->
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():  

    # Load documents ->
    documents = []
    for root, _, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading file: {file}")
                loader = PyPDFLoader(os.path.join(root, file))
                documents.extend(loader.load())

    # Create vector store ->
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))    
    
    # Create chain
    llm = llm_pipeline()
    retriever = vectorstore.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(llm,
                                          retriever)
    return qa

def process_answer(question):

    # Get answer with RAG ->
    qa = qa_llm()
    generated_text = qa.invoke({"question": question, "chat_history": chat_history})
    answer = generated_text['answer']
    return answer

def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.title('Chatbot for PDF Data')
    user_input = st.text_input("Question about your data:", key="input")

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Active!"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hello!"]
        
    if user_input:
        answer = process_answer(user_input)
        chat_history.extend([HumanMessage(content=user_input), answer])

        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(answer)

    if st.session_state["generated"]:
        display_conversation(st.session_state)

if __name__ == '__main__':
    main()
