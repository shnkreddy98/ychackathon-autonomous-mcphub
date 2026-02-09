import asyncio
import sys

from services.llm import Agent


async def main(question: str):
    agent = Agent()
    res = await agent.run(question, max_iterations=25)

    print(f"Output: {res.output}")
    print(f"Usage: {res.usage}")


if __name__ == "__main__":
    question = sys.argv[1]
    asyncio.run(main(question))
