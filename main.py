import os
import json
from dotenv import load_dotenv
from orchestrator import ResearchAssistant

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set in environment")
    if not os.getenv("SERPAPI_API_KEY"):
        raise ValueError("SERPAPI_API_KEY not set in environment")

    question = input("Enter your question: ")
    assistant = ResearchAssistant(max_results = 10)
    result = assistant.answer(question)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()