import sys
import base64

from langchain.tools import tool
from langchain import hub
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # 将父级目录加入执行目录列表

from agent.tools import load_tool
from agent.prompts import chat_prompt
from llms.zhipuai_llm import ZhipuAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv
from langchain.agents import (
    create_structured_chat_agent,
    AgentExecutor,
    Agent
)
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.chains.conversation.memory import ConversationSummaryMemory
import os

from langchain.chains.llm import LLMChain
# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
api_key = os.environ["ZHIPUAI_API_KEY"]

def llm_test_1():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-3-turbo",
    )

    #my_prompt = hub.pull("hwchase17/structured-chat-agent")
    my_prompt = chat_prompt


    tools = load_tool()

    agent = create_structured_chat_agent(
        llm = zhipuai_chat,
        tools=tools,
        prompt= my_prompt
    )



    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat)

    agent_executor = AgentExecutor(agent=agent,tools=tools,memory=summary_mem,verbose=True,handle_parsing_errors=True)

    try:
        answer = agent_executor.invoke(
            {
                "objective":"Search pear for me.",
                "marked_screenshot":"",
                "bound_list":"[(index:1,text:'',tab:Search),(index:2,text:'Bing',tab:Button)]",
                "chat_history":f'{summary_mem.load_memory_variables({})}'
            }
        )
        print(answer)
    except Exception:
        exit(0)

def llm_test_2():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-3-turbo",
    )

    my_prompt = hub.pull("hwchase17/structured-chat-agent")
    #my_prompt = PromptTemplate(template="You are a helpful assistant.You are expected to answer my question: '{input}'",input_variables=["input"])
    tools = load_tool()

    agent = create_structured_chat_agent(
        llm = zhipuai_chat,
        tools=tools,
        prompt= my_prompt
    )



    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat,memory_key='chat_history',input_key='human_input')

    agent_executor = AgentExecutor(agent=agent,tools=tools,memory=None,verbose=True,handle_parsing_errors=True,return_intermediate_steps=True)

    answer = agent_executor.invoke(
        {
            "input":"3*5=?"
        }
    )
    print(answer)

def llm_test_3():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-3-turbo",
    )

    my_prompt = chat_prompt
    #my_prompt = PromptTemplate(template="You are a helpful assistant.You are expected to answer my question: '{input}'",input_variables=["input"])
    tools = load_tool()

    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat,memory_key='chat_history',input_key='human_input')

    #agent_executor = AgentExecutor(agent=agent,tools=tools,memory=summary_mem,verbose=True,handle_parsing_errors=True)
    

    agent_executor = LLMChain(
        llm= zhipuai_chat,
        memory=summary_mem,
        prompt=my_prompt,
        verbose=True,
    )
    response = agent_executor.invoke(
        input=
        {
            "objective":"Search pear for me.",
            "marked_screenshot":"",
            "bound_list":"[(index:1,text:'',tab:Search),(index:2,text:'Bing',tab:Button)]",
            "chat_history":f"{summary_mem.load_memory_variables({})}",
            "human_input":""
        }
    )
    print(response)
    print(type(response))
    print(summary_mem.load_memory_variables({}))

def llm_test_4():
    zhipuai_chat = ZhipuAI(
        temperature=0.5,
        api_key=api_key,
        model_name="glm-3-turbo",
    )

    my_prompt = chat_prompt
    #my_prompt = PromptTemplate(template="You are a helpful assistant.You are expected to answer my question: '{input}'",input_variables=["input"])
    tools = load_tool()

    summary_mem = ConversationSummaryMemory(llm = zhipuai_chat,memory_key='chat_history',input_key='human_input')


    

    



#print(summary_mem.load_memory_variables({}))
if __name__ == "__main__":
    llm_test_3()
"""
with open('test.png','rb') as f:
    base64_data = base64.b64encode(f.read())




agent = initialize_agent(tools = tools,llm=zhipuai_chat, verbose=True)

message = messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "如果让你在这个页面搜索'langchain'，你会干什么？"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{base64_data}"
                    }
                }
            ]
        }
]

res = agent.run(
    "who are u?"
)
print(res)
"""