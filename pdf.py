from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import os
from langchain.chains import RetrievalQA
import gradio as gr
embeddings = FakeEmbeddings(size=1324)
llm = HuggingFacePipeline.from_model_id(model_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-AWQ",task="text-generation",device=0,pipeline_kwargs={"max_new_tokens":300})
text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
def chatbot(file,query):
    loader=UnstructuredFileLoader(file)
    documents = loader.load()
    docs= text_splitter.split_documents(documents)
    db=FAISS.from_documents(docs,embeddings)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=db.as_retriever(),
                                    input_key="question")
    final=chain.invoke(query)
    return final['result']
iface = gr.Interface(
    fn=chatbot,
    inputs=["file","text"],
    outputs="text",
    title="PDF chatbot",
    ).launch()
