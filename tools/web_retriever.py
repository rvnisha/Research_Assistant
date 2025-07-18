import os
from langchain_community.utilities import SerpAPIWrapper
from langchain.schema import Document

class WebRetriever:
    def __init__(self, max_results: int = 5):
        api_key = os.getenv("SERPAPI_API_KEY")
        self.search = SerpAPIWrapper()
        self.max_results = max_results

    def retrieve(self, query: str):
        # Perform a web search via SerpAPI
        result_text = self.search.run(query)
        # Split into lines/snippets
        snippets = result_text.split("\n")[: self.max_results]
        docs = []
        for snippet in snippets:
            text = snippet.strip()
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={"source": "serpapi_search"}
            ))
        return docs