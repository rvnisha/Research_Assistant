from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class Summarizer:
    def __init__(self):
        # Chat template for summarization
        system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

        research_template = """
           Information:
            --------
            {context}
            --------
            Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
            The report should focus on the answer to the question, should be well structured, informative, \
            in depth, with facts and numbers if available and a minimum of 1,200 words.
            You should strive to write the report as long as you can using all relevant and necessary information provided.
            You must write the report with markdown syntax.
            You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
            Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
            You must write the report in apa format.
        """
        self.prompt = ChatPromptTemplate.from_messages(
            [
             ("system", system_prompt),
             ("user", research_template),
            ]
        )
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
        self.chain = self.prompt | self.llm

    def summarize(self, question: str, insights: list):
        # Format insights into a context block
        context = "\n".join(f"- {ins}" for ins in insights)
        # Invoke the chain with question and context
        result = self.chain.invoke({"question": question, "context": context})
        return result.content.strip()