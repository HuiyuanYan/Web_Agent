from langchain.tools import BaseTool
from langchain.chains import LLMMathChain
import numexpr
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import(
    Type,Optional
)
import asyncio

class MathCalculationTool(BaseTool):
    name: str = "math_calculation"
    description: str = (
        "This tool harnesses the power of the NumExpr library to perform lightning-fast mathematical calculations. "
        "It supports a wide range of mathematical functions, including basic arithmetic operations, trigonometric functions, "
        "exponential and logarithmic functions, and more. "
        "To use this tool effectively, simply provide a mathematical expression as input."
    )
    def _run(
        self,
        expr: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Perform mathematical calculation using NumExpr."""
        result = numexpr.evaluate(expr)
        if run_manager:
            run_manager.on_tool_end(result)
        return result
        
    async def _arun(
        self, 
        expr: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Perform mathematical calculation asynchronously."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None,numexpr.evaluate, expr)
        if run_manager:
            run_manager.on_tool_end(result)
        return result