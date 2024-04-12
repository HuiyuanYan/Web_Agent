from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import(
    Type,Optional
)
import asyncio

class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )

class WikipediaQueryTool(BaseTool):
    name :str= "wikipedia_query"
    description:str =str(
        "This tool provides a convenient way to search and retrieve information from Wikipedia. "
        "It is useful for answering general questions about people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = WikiInputs
    
    def __init_subclass__(cls, **kwargs: WikipediaQueryRun) -> None:
        cls.api_wrapper = WikipediaAPIWrapper(
            doc_content_chars_max = 1000,
            top_k_results = 1
        )
        return super().__init_subclass__(**kwargs)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia tool."""
        
        result = self.api_wrapper.run(query)
        if run_manager:
            run_manager.on_tool_end(result)
        return result


        pass
    async def _arun(
        self, 
        query: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.api_wrapper.run, query)
        if run_manager:
            run_manager.on_tool_end(result)
        return result