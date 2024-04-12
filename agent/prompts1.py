from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    BaseChatPromptTemplate
)
from typing import (
    Any,
    List
)
from langchain.tools import Tool
from langchain_core.messages import BaseMessage

from langchain.output_parsers import(
    StructuredOutputParser,
    ResponseSchema
)



system_message_template = """
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
The user's objective: This is the task you're trying to complete.
The current web page's marked screenshot with interactive elements: the second picture is current webpage screenshot highlights interactive elements on the webpage with numbered rectangles, indicating areas for clicking, typing, or hovering to interact with key information.Please look at the screenshot carefully,Try to make best decision every step.
The current webpage's bound_list:The bound_list is a structured association that connects visually selectable elements, which are highlighted by numbered rectangular frames in the screenshot. These elements could be select dropdowns or any other type of web page elements. When a specific interaction is required with any of these elements, you should reference the bound index indicated in the image to directly interact with the respective elements. Each entry in the boundMap comprises:
\tindex: A unique identifying number assigned to an element within the screenshot, which users can reference to locate and interact with the specific element on the page.
\ttext: The actual textual list of all options or information displayed within the element, valuable when you need to select a particular option or interact with an obscured element.\n\tCategory: The category of the element such as 'SELECT', 'BUTTON', 'INPUT', 'TEXT', and others. This helps the intelligent system to determine what kind of operation to perform on the element.
\ttype: The category of the element such as 'SELECT', 'BUTTON', 'INPUT', 'TEXT', and others. This helps the intelligent system to determine what kind of operation to perform on the element.
The previous action: This is the action you just performed. It may be helpful to track your progress.
The current web page's URL: This is the page you're currently navigating.
The actions you can perform fall into several categories:
- Page Operation Actions:
`click [id]`:This action clicks on an element using its index from the screenshot.
`type [id] [content] [press_enter_after=0|1]`:The 'type' action clears the current content of the input field associated with the given index before inputting the new 'content'. By default, it triggers the 'Enter' key after typing, unless 'press_enter_after' is set to 0.
`hover [id]`:This action hovers over an element using its index from the boundMap.
`press [key_comb]`:Simulates the pressing of a key combination on the keyboard (e.g., Ctrl+v).
`scroll [direction=down|up|left|right]`:Scroll the page up or down,left or right.such as 'scroll [up]' or 'scroll [down]'.

Tab Management Actions:
`new_tab`:Open a new, empty browser tab.
`tab_focus [tab_index]`:Switch the browser's focus to a specific tab using its index.
`close_tab`:Close the currently active tab.

URL Navigation Actions:
`goto [url]`:Navigate to a specific URL.
`go_back`:Navigate to the previously viewed page.
`go_forward`:Navigate to the next page .

Completion Action:
`stop [answer]`:Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete or find the answer from existing webpages, provide the answer as \"N/A\" in the bracket.

To be successful, it is very important to follow the following rules:
1. You should only issue an action that is valid given the current screenshot
2. You should only issue one action at a time.
3. You should start you thinking with a detailed reasoning using 'Let's consider this step-by-step. I think...', explaining the reasoning process. Then, it should include 'In summary, the next action I will perform is', introducing the specific action enclosed in backticks.
4. Issue stop action when you think you have achieved the objective.
You can use the following tools for the tasks:
{tools}

All of the actions you take must follow the format:
{format_instructions}


Here are some examples:

If you are asked to complete the following web task and provide the following information:
User's Objective: "Tell me who the founder of Apple is",
Current URL: `https://baidu.rudon.cn/`
Bound List:[(index=0, text='', tag='INPUT')(index=1, text='', tag='INPUT')(index=2, text='【留言板】', tag='A')(index=3, text='More', tag='A')(index=4, text='百度清爽版', tag='SPAN')(index=5, text='Today 522', tag='DIV')],
Previous Chat History: None
You must think and give actions in the following format:
```
{{
Thought: "Let's think step by step,my task is to find the founder of Apple, and I need to carefully review the screenshots to gather information. From the screenshot, it can be seen that there is a marked search box in the middle with an index of 0. In the bound list, it is indicated that the attribute of this element (with index 1) is "OUT", so I can use the "TYPE" action in the rules to search for the founder of Apple.In summary, the next action i preform is 'type[0][the founder of Apple][1]'.",
Action: "type[0][the founder of Apple][1]",
Tool_Call: "none",
Tool_Args:"none"
}}
```
Please remember to answer the question in the format provided in the example and provide the answer in the form of a JSON string!

"""

human_message_template = [
    {
        "type":"text",
        "text": 
                "Objective: {objective}\n"
                "Current URL: {current_url}\n"
                "Bound List: {bound_list}\n"
                "Previous Action: {previous_action}\n"
                "Previous Chat History: {chat_history}\n"
                "{human_input}"
    },
    {
        "type":"image_url",
        "image_url":"{image_url}"
    },   
]

system_message_prompt = SystemMessagePromptTemplate.from_template(template = system_message_template)

human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_message_template)

chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,human_message_prompt]
    )





if __name__ == "__main__":
    response_schemas = [
            ResponseSchema(name="Thought",description="give your thoughts based on the current task scenario"),
            ResponseSchema(name="Action", description="the action you want to take must be one of the actions specified above, or it can be empty when you want to invoke the tool"),
            ResponseSchema(name="Tool Call", description="when you decide that you want to call a tool, fill in the tool name, otherwise leave it blank"),
            ResponseSchema(name="Tool Args",description="when you want to call the tool, give the argument in the form of json, otherwise it is not filled")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schemas)
    
    chat_prompt = ChatPromptTemplate(
        messages=[system_message_prompt,human_message_prompt],
        input_variables=["objective","bound_list","chat_history","human_input","tools","tool_names","image_url"],
        partial_variables={"format_instructions":''}
    )
    assert "format_instructions" in chat_prompt.partial_variables
    chat_prompt.partial_variables['format_instructions'] = output_parser.get_format_instructions()
    print(output_parser.parse("""
        {
        "Thought": "I believe it's important to understand the task scenario thoroughly before proceeding with any action. In this case, the task requires generating a markdown code snippet according to a specific schema. It's crucial to ensure that the output meets the required formatting and content criteria.",
        "Action": "Generate the markdown code snippet as requested, ensuring proper indentation and syntax.",
        "Tool Call": "cal",
        "Tool Args": ""
        }
    """)["Tool Call"]==None)

    print(chat_prompt.partial_variables)
    print(dict(chat_prompt.invoke(
        {
            "objective":"Search pear for me.",
            "bound_list":"[(index:1,text:'',tab:Search),(index:2,text:'Bing',tab:Button)]",
            "chat_history":f"",
            "human_input":"",
            "tools":"",
            "tool_names":"",
            "image_url":"BASE64",
        })
    ))


    