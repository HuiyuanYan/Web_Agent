# coding=utf-8
import sys

from langchain.tools import tool
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表
from langchain import hub
from agent.tools import load_tool
from llms.zhipuai_llm import ZhipuAI
from langchain.agents import initialize_agent
from langchain.prompts import(ChatPromptTemplate,HumanMessagePromptTemplate,SystemMessagePromptTemplate)
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
import base64
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]



def png_bytes_to_base64(png: bytes) -> str:
    """
    Convert PNG image bytes to Base64 encoded string

    Args:
    png (bytes): Bytes representation of the PNG image

    Returns:
    str: Base64 encoded string
    """
    base64_encoded = base64.b64encode(png)
    base64_string = base64_encoded.decode('utf-8')
    return base64_string



def llm_test_1():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-4v",
    )
    template2 = [
        {
            "type":"text",
            "text":"Chat History:{chat_history}"
                    "Question:{input}"
                    "{human_input}"
        },
        {
            "type":"image_url",
            "image_url":"{image_url}"
        }  
    ]
    template1 = "You are a helpful assistant."
    prompt =ChatPromptTemplate.from_messages(
                    [
                        SystemMessagePromptTemplate.from_template(template=template1),
                        HumanMessagePromptTemplate.from_template(template=template2)
                    ]
    )
    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat,memory_key='chat_history',input_key='human_input')

    chain = LLMChain(
        llm = zhipuai_chat,
        prompt=prompt,
        verbose=True,
        memory=summary_mem
    )

    with open('marked_screenshot_0.png','rb')as f:
        img = f.read()
    
    answer = chain.invoke(
        {
            "chat_history":f"{summary_mem.load_memory_variables({})}",
            "input":"给出了你一张网页截图，你从截图中看到了什么？以及图中被标记的元素共有几个？",
            "image_url":png_bytes_to_base64(img),
            "human_input":""
        }
    )

    
    print(answer)
    print(summary_mem.load_memory_variables({}))


    



#print(summary_mem.load_memory_variables({}))
if __name__ == "__main__":
    llm_test_1()
