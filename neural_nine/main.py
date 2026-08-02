import asyncio

from llama_index.core.workflow.context import Context

from src.agents.rag.agent import get_agent as rag_agent
from src.agents.color.agent import get_agent as color_agent
# from src.tools.storage.tool import ask_question

# Messages
PROMPT_CHOICE = 'Do you want color/rag/storage?: '
PROMPT_QUESTION = 'What is your question?: '
MSG_CHAT_START = 'You can now chat with the agent. Type "exit" to quit.'
MSG_EXIT = 'Exiting chat...'
MSG_INVALID_CHOICE = 'Invalid choice. Please enter "color" or "chat".'


async def async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


async def main():
    choice = (await async_input(PROMPT_CHOICE)).strip().lower()

    if choice == 'rag_default':
        agent = rag_agent()
        ctx = Context(agent)
        print(MSG_CHAT_START)
        while True:
            question = await async_input(PROMPT_QUESTION)
            if question == 'exit':
                print(MSG_EXIT)
                break
            response = await agent.run(question, ctx=ctx)
            print(response)
    elif choice == 'storage':
        print('No supported yet...')
        # print(MSG_CHAT_START)
        # while True:
        #     question = await async_input(PROMPT_QUESTION)
        #     if question == 'exit':
        #         print(MSG_EXIT)
        #         break
        #     response = ask_question(question)
        #     print(response)
    elif choice == 'color':
        agent = color_agent()
        ctx = Context(agent)
        print(MSG_CHAT_START)
        while True:
            question = await async_input(PROMPT_QUESTION)
            if question == 'exit':
                print(MSG_EXIT)
                break
            response = await agent.run(question, ctx=ctx)
            print(response)
    else:
        print(MSG_INVALID_CHOICE)

if __name__ == '__main__':
    asyncio.run(main())
