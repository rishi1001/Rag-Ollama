from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from VectorDB.Chorma import Retriever

class ChatPDF:
    def __init__(self):
        self.model = ChatOllama(model="llama3", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise </s> 
            Question: {question} 
            Context: {context} 
            Answer: 
            """
        )
        self.retriever = Retriever()
        self.vector_store = None
        self.chain = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF document by extracting its content, splitting it into chunks,
        and saving it to a vector database.
        """
        documents = self.retriever.get_documents(pdf_file_path)
        self.retriever.save_vectordb_locally(documents=documents)

    def ask(self, query: str) -> str:
        """
        Process a query by retrieving relevant context and generating an answer using
        the language model.
        """
        context = self.retriever.rag()
        self.chain = ({"context": context, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
        answer = self.chain.invoke(query)
        return answer

    def clear(self):
        """
        Clear the internal state by resetting the vector store, retriever, and chain.
        """
        self.vector_store = None
        self.retriever = None
        self.chain = None
