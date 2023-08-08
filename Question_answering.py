from typing import List
from functools import lru_cache
from typing import Any, Dict
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

import logging
import os
import sys

# current_dir = os.getcwd()
# parent_dir_two_levels_back = os.path.dirname(os.path.realpath(current_dir))
# if parent_dir_two_levels_back not in sys.path:
#     sys.path.append(parent_dir_two_levels_back)
# print(parent_dir_two_levels_back)

log_format = '%(asctime)s:%(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


def question_answering(api_key_val: str, question: str, index: Any) -> str:
    qa = RetrievalQA.from_chain_type(llm=OpenAI(
        openai_api_key=api_key_val), chain_type="map_reduce", retriever=index.as_retriever())
    tools = [Tool(name="State of Union QA System", func=qa.run,
                  description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.")]
    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                You have access to a single tool:"""
    suffix = """Begin!
            {chat_history}
            Question: {input}
            {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    if "memory" not in question_answering.__dict__:
        question_answering.memory = ConversationBufferMemory(
            memory_key="chat_history")

    llm_chain = LLMChain(
        llm=OpenAI(temperature=0, openai_api_key=api_key_val,
                   model_name="gpt-3.5-turbo"),
        prompt=prompt,
    )
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=question_answering.memory)
    # Perform the question answering
    res = agent_chain.run(question)
    return res
