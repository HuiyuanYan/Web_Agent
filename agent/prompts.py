from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)

system_message_template = """
You are an autonomous intelligent agent tasked with navigating a web browser. You will be given web-based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Here's the information you'll have:
Objective
Marked Screenshot
Bound List
The actions you can perform fall into several categories:
`click [id]`
`type [id] [content] [press_enter_after=0|1]`
`hover [id]`
`press [key_comb]`
`scroll [direction=down|up|left|right]`

Tab Management Actions:
`new_tab`
`tab_focus [tab_index]`
`close_tab`

URL Navigation Actions:
`goto [url]`
`go_back`
`go_forward`

Completion Action:
`stop [answer]`
You can only issue one action at a time.


All of the actions you take must follow the format:
Thought:
Action:



Here are some examples:

Given OBSERVATION as follows:
Objective: "Search term 'apple' for me.",
Marked Screenshot: "None",
Bound List:"marked element: 1-'bing'-Button, ..., 17-''-Search, ...",

You must think and take only one action as follows:

Thought: "The element numbered 17 appears to be the search box, so I need to use it to search."
Action: "type[17][apple][0]"
"""


human_message_template = """
Objective: {objective}
Marked Screenshot(base64-encoded): {marked_screenshot}
Bound List: {bound_list}
Previous Chat History: {chat_history}
"""


system_message_prompt = SystemMessagePromptTemplate.from_template(template = system_message_template)

human_message_prompt = HumanMessagePromptTemplate.from_template(template=human_message_template)


chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt,human_message_prompt]
    )

if __name__ == "__main__":
    objective = "Search 'Pear' for me."
    marked_screenshot = ""
    chat_history = ""
    