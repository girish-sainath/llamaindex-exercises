import asyncio
from dotenv import load_dotenv

from llama_index.core.agent.workflow import ReActAgent as ReActAgentWorkflow
from llama_index.core.agent import ReActAgent as ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.litellm import LiteLLM
from llama_index.core.tools import FunctionTool


load_dotenv()


def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers and returns the result.
    :param a: First integer.
    :param b: Second integer.
    :return: The product of a and b.
    """
    return a * b


def add(a: int, b: int) -> int:
    """
    Adds two integers and returns the result.
    :param a: First integer.
    :param b: Second integer.
    :return: The sum of a and b.
    """
    return a + b


def subtract(a: int, b: int) -> int:
    """
    Subtracts the second integer from the first and returns the result.
    :param a: First integer.
    :param b: Second integer.
    :return: The difference of a and b.
    """
    return a - b


def divide(a: int, b: int) -> float:
    """
    Divides the first integer by the second and returns the result.
    :param a: First integer.
    :param b: Second integer.
    :return: The quotient of a and b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


async def execute_agent() -> None:
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    subtract_tool = FunctionTool.from_defaults(fn=subtract)
    division_tool = FunctionTool.from_defaults(fn=divide)

    agent = ReActAgent(
        llm=LiteLLM(
            model="sap/anthropic--claude-4.6-sonnet",
        ),
        tools=[
            multiply_tool,
            add_tool,
            subtract_tool,
            division_tool,
        ],
        system_prompt="You are a helpful assistant that can perform basic arithmetic operations. Use the provided tools to answer questions about addition, subtraction, and multiplication.",
    )

    prompt: str = "What is 6 * 3 / 3 + 4 - 10?"
    print(f"Prompt: {prompt}\n")
    response = await agent.run(prompt)
    print(f"\nAnswer: {response}")


async def execute_agent_workflow() -> None:
    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    subtract_tool = FunctionTool.from_defaults(fn=subtract)
    division_tool = FunctionTool.from_defaults(fn=divide)

    agent = ReActAgentWorkflow(
        llm=LiteLLM(
            model="sap/anthropic--claude-4.6-sonnet",
        ),
        tools=[
            multiply_tool,
            add_tool,
            subtract_tool,
            division_tool,
        ],
        system_prompt="You are a helpful assistant that can perform basic arithmetic operations. Use the provided tools to answer questions about addition, subtraction, and multiplication.",
    )

    ctx = Context(agent)

    prompt: str = "What is 6 * 3 / 3 + 4 - 10?"
    print(f"Prompt: {prompt}\n")
    response = await agent.run(prompt, ctx=ctx)
    print(f"\nAnswer: {response}")


async def main():
    print("Executing ReActAgent...\n")
    await execute_agent()
    print("\nExecuting ReActAgentWorkflow...\n")
    await execute_agent_workflow()


if __name__ == "__main__":
    asyncio.run(main())
