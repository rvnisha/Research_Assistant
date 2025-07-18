from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class Extractor:
    def __init__(self):
        template = """
        Extract the key insight from the following web snippet:

        {text}
  
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        # Initialize the LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106",temperature=0)
        # Chain the chat prompt to the LLM
        self.chain = self.prompt | self.llm

    def extract(self, document):
        # Invoke the chain with the snippet text
        result = self.chain.invoke({"text": document.page_content})
        return result.content.strip()