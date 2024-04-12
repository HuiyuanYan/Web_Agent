# coding=utf-8
import sys


from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表

from agent.tools import load_tool
from agent.agent import WebAgent
from agent.prompts1 import chat_prompt
from agent.tools import load_tool
from browser_env.env import BrowserEnv

from llms.zhipuai_llm import ZhipuAI

from dotenv import find_dotenv, load_dotenv

from langchain.chains.conversation.memory import ConversationSummaryMemory
import os


# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]


 

def web_agent_test_1():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-4v",
    )
    browser_env = BrowserEnv(
            headless=False,
            slow_mo=1000,
            viewport_size={"width":640,"height":360}
    )

    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat,memory_key='chat_history',input_key='human_input')

    agent = WebAgent(
        llm = zhipuai_chat,
        prompt=chat_prompt,
        browser_env=browser_env,
        tools= load_tool(),
        memory=summary_mem
    )
    agent.finish_web_task(
        "https://ditu.amap.com/",
        "颐和园在哪里？",
        max_iterations=7,
        trace_dir='results/result_3'
    )

if __name__ == "__main__":
    web_agent_test_1()
    


    

