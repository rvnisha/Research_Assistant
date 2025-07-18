import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class DocRetriever:
    """
    RAG over a set of PDF files:
      - ingest(file_paths): load & index PDFs in‑memory
      - retrieve(query): run similarity search and return top‑k chunks
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_results: int = 5
    ):
        self.chunk_size   = chunk_size
        self.chunk_overlap= chunk_overlap
        self.max_results  = max_results
        self.embeddings   = OpenAIEmbeddings()
        self.vectorstore  = None

    def ingest(self, file_paths: List[str]):
        #Load all PDFs
        docs = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        # Storing the embeddings in FAISS vectorstore
        self.vectorstore = FAISS.from_documents(
            chunks,
            self.embeddings,
        )

    def retrieve(self, query: str) -> List[Document]:
        if self.vectorstore is None:
            raise ValueError("DocRetriever: call ingest(...) first with your PDF paths.")
        ##Prompt for report
        system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

        research_template = """
           Information:
            --------
            {context}
            --------
            Using the above information, answer the following question or topic: "{input}" in a detailed report -- \
            Please conduct a comprehensive literature review that focuses on identifying seminal works, major theoretical frameworks, research methodologies\
            Summarize the key findings without missing any important information that would be needed in my research
            The report should focus on the answer to the question, should be well structured, informative, \
            in depth, with facts and numbers if available and a minimum of 1,200 words.
            You must write the report with markdown syntax.
            You must write the report in apa format.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
             ("system", system_prompt),
             ("user", research_template),
            ]
        )

        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)

        self.retriever = self.vectorstore.as_retriever(search_type = "similarity", search_kwargs = {'k':self.max_results})
        self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        self.response = self.rag_chain.invoke({"input": query})
        #Source retrival
        self.sources = [doc.metadata['source'] for doc in self.response['context']]
        return self.response['answer'], self.sources
