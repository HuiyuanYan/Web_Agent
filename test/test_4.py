# coding=utf-8
import sys

from langchain.tools import tool
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
from langchain import hub
from agent.tools import load_tool
from llms.zhipuai_llm import ZhipuAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor
)
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.tools import (
    BaseTool
)
from langchain.chains.conversation.memory import ConversationSummaryMemory
import os

from langchain.chains.llm import LLMChain
from langchain.agents import ConversationalChatAgent
# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]

from typing import Type
 
class TaskTool(BaseTool):
    name = "return a task and its description,you must use"
    description :str= str("when you need to finish a task, use this tool")

    def _run(self) -> dict:
        return {
            "task":"Slove the question: 3*5.",
            "description":"just answer my question"
        }
    
    async def _arun(self, query: str) -> str:
        raise NotImplementedError("ERROR")


def llm_test_1():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-3-turbo",
    )

    my_prompt = hub.pull("hwchase17/structured-chat-agent")


    tools = [TaskTool()]

    agent = create_structured_chat_agent(
        llm = zhipuai_chat,
        tools=tools,
        prompt= my_prompt
    )



    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat)


    agent_executor = AgentExecutor(agent=agent,tools=tools,memory=summary_mem,verbose=True,handle_parsing_errors=True)
    answer = agent_executor.invoke({"input":"Use the tool i provide to you."})
    print(answer)


    



#print(summary_mem.load_memory_variables({}))
if __name__ == "__main__":
    llm_test_1()
