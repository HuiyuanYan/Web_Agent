from .math_tool import MathCalculationTool
from .wikipedia_tool import WikipediaQueryTool

__all__ = ["load_tool"]

def load_tool()->list:
    return [MathCalculationTool(),WikipediaQueryTool()]
