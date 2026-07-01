import re

MAX_STEPS = 6

SYSTEM_PROMPT = """
You are a Python AI assistant.

You solve problems by reasoning step-by-step.

You may use these tools.

retrieve_documents(question)

- searches the beginner Python tutorial
- use for:
  variables
  loops
  functions
  classes
  inheritance
  exceptions
  lists
  tuples
  dictionaries
  OOP
  beginner concepts

search_python_docs(question)

- searches docs.python.org
- use for:
  pathlib
  typing
  itertools
  functools
  asyncio
  dataclasses
  collections
  contextlib
  standard library
  advanced builtins
  language reference

Rules

1. Think before acting.
2. Use a tool only when needed.
3. Call ONLY ONE tool per response.
4. Wait for the Observation before continuing.
5. Never invent tool results.
6. Stop only when you have enough information.
7. Never invent tool names.

Use exactly one of these formats.

Thought: your reasoning

Action: tool_name("question")

OR

Final Answer: your answer
"""


class Agent:

    def __init__(self, client):

        self.client = client
        self.tools = {}

    def register_tool(self, name, func):

        self.tools[name] = func

    def ask_llm(self, conversation):

        response = self.client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=conversation,
            temperature=0.2,
            max_tokens=500,
        )

        return response.choices[0].message.content

    def parse_action(self, text):

        match = re.search(
            r'Action:\s*(\w+)\("([^"]+)"\)',
            text,
            re.DOTALL
        )

        if not match:
            return None, None

        return match.group(1), match.group(2)

    def run(self, question):

        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        for step in range(MAX_STEPS):

            print(f"\n========== Step {step + 1} ==========")

            llm_response = self.ask_llm(conversation)

            print("\n==========================")
            print("LLM RESPONSE")
            print("==========================\n")
            print(llm_response)

            if "Final Answer:" in llm_response:

                return llm_response.split(
                    "Final Answer:",
                    1
                )[1].strip()

            tool_name, tool_input = self.parse_action(
                llm_response
            )

            if tool_name is None:

                return (
                    "The model did not produce a valid tool call "
                    "or final answer.\n\n"
                    f"Response:\n{llm_response}"
                )

            if tool_name not in self.tools:

                return f"Unknown tool: {tool_name}"

            print("\n==========================")
            print("TOOL REQUEST")
            print("==========================")
            print(f"Tool : {tool_name}")
            print(f"Input: {tool_input}")

            while True:

                approval = input(
                    "\nApprove tool execution? (y/n): "
                ).strip().lower()

                if approval in ("y", "yes"):

                    break

                if approval in ("n", "no"):

                    return "Tool execution cancelled by user."

                print("Please enter 'y' or 'n'.")

            print(f"\nRunning Tool: {tool_name}\n")

            observation = self.tools[tool_name](
                tool_input
            )

            conversation.append(
                {
                    "role": "assistant",
                    "content": llm_response,
                }
            )

            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation from {tool_name}:\n\n"
                        f"{observation}\n\n"
                        "Based on this observation:\n"
                        "- Either call ONE more tool if necessary.\n"
                        "- Or provide the Final Answer.\n"
                        "Do not invent observations."
                    ),
                }
            )

        return (
            f"Agent stopped after {MAX_STEPS} reasoning steps "
            "without producing a Final Answer."
        )