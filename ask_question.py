import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from agent import Agent
from tools import (
    retrieve_documents,
    retrieve_python_docs,
)

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)


def main():

    question = input("\nAsk a question: ")

    agent = Agent(client)

    agent.register_tool(
        "retrieve_documents",
        retrieve_documents,
    )

    agent.register_tool(
        "search_python_docs",
        retrieve_python_docs,
    )

    answer = agent.run(question)

    print("\n========================")
    print("FINAL ANSWER")
    print("========================\n")

    print(answer)


if __name__ == "__main__":
    main()