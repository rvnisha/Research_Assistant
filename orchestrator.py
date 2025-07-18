from tools.web_retriever import WebRetriever
from tools.extractor import Extractor
from tools.summarizer import Summarizer

class ResearchAssistant:
    def __init__(self, max_results: int = 5):
        # Initializes the SerpAPI-based web retriever
        self.web_retriever = WebRetriever(max_results=max_results)
        # Initializes extraction and summarization modules
        self.extractor = Extractor()
        self.summarizer = Summarizer()

    def answer(self, question: str) -> dict:
        # 1. Retrieve web snippets
        web_docs = self.web_retriever.retrieve(question)

        # 2. Extract main insights from each snippet (considering token limit)
        insights = [self.extractor.extract(doc) for doc in web_docs]

        # 3. Summarize into final answer
        answer = self.summarizer.summarize(question, insights)

        # Collect sources (URLs)
        sources = [doc.metadata.get("source") for doc in web_docs]

        return {"question": question, "answer": answer, "sources": sources}