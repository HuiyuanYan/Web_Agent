from enum import IntEnum
from abc import abstractmethod
from playwright.async_api import (
    Page as APage,
    BrowserContext as ABrowserContext
)
from playwright.sync_api import (
    Page,
    BrowserContext
)
from beartype.typing import(
    TypedDict,
    List
)
from beartype import beartype
from itertools import chain
import re
from .utils import MarkedElement
from .constant import(
    SPECIAL_KEY_MAPPINGS,
    ASCII_CHARSET,
    FREQ_UNICODE_CHARSET,
    CHINESE_CHARSET,
    SPECIAL_KEYS
)
from .processor import ObservationProcessor

class ActionTypes(IntEnum):
    """Valid action types for browser env."""

    NONE = 0
    # mouse wheel and keyboard, universal across all action spaces
    SCROLL = 1
    KEY_PRESS = 2

    # low level mouse and keyboard actions

    # mid level mouse and keyboard actions
    CLICK = 6
    TYPE = 7
    HOVER = 8

    # page level actions, universal across all action spaces
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14

    # high-leval actions that playwright support
    STOP = 17 #TODO

    def __str__(self) -> str:
        return f"ACTION_TYPES.{self.name}"
    
class Action(TypedDict):
    action_type: int
    text: list[int]
    direction: str
    key_bomb: str
    answer:str
    page_number:int
    mark_id: int


_key2id: dict[str, int] = {
            key: i
            for i, key in enumerate(
                chain(SPECIAL_KEYS, ASCII_CHARSET, FREQ_UNICODE_CHARSET, ["\n"])
            )
        }
_id2key: list[str] = sorted(_key2id, key=_key2id.get)

class ActionHandler:

    @staticmethod
    @beartype
    def _keys2ids(keys: list[int | str] | str) -> list[int]:
        return list(
            map(
                lambda key: _key2id[str(key)]
                if isinstance(key, str)
                else int(key),
                keys,
            )
        )
    
    @staticmethod
    @beartype
    def _create_none_action()->Action:
        return{
            "action_type":ActionTypes.NONE,
            "text":[],
            "direction":"",
            "key_bomb":"",
            "answer":"",
            "page_number":0,
            "mark_id":""
        }
    
    @classmethod
    @beartype
    def _create_stop_action(cls,answer:str)->Action:
        action = cls._create_none_action()
        action.update(
            {
                "action_type":ActionTypes.STOP,
                "answer":answer
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_scroll_action(cls,direction:str)->Action:
        assert direction in ["up","down","left","right"]
        action = cls._create_none_action()
        action.update(
            {
                "action_type":ActionTypes.SCROLL,
                "direction":direction            
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_key_press_action(cls,key_comb: str) -> Action:
        """Return the key press action"""

        def map_keys(key_comb: str) -> str:
            keys = key_comb.split("+")
            mapped_keys = []
            for key in keys:
                mapped_key = SPECIAL_KEY_MAPPINGS.get(key.lower(), key)
                mapped_keys.append(mapped_key)
            return "+".join(mapped_keys)
        action = cls._create_none_action()
        mapped_key_comb = map_keys(key_comb)
        action.update(
            {
                "action_type": ActionTypes.KEY_PRESS,
                "key_comb": mapped_key_comb
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_page_focus_action(cls,page_number: int) -> Action:
        """Return a valid action object with type PAGE_FOCUS."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.PAGE_FOCUS,
                "page_number": page_number,
            }
        )
        return action

    @classmethod
    @beartype
    def _create_new_tab_action(cls) -> Action:
        """Return a valid action object with type NEW_TAB."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.NEW_TAB,
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_go_back_action(cls) -> Action:
        """Return a valid action object with type GO_BACK."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.GO_BACK,
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_go_forward_action(cls) -> Action:
        """Return a valid action object with type GO_FORWARD."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.GO_FORWARD,
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_goto_url_action(cls,url: str) -> Action:
        """Return a valid action object with type GOTO_URL."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.GOTO_URL,
                "url": url,
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_page_close_action(cls) -> Action:
        """Return a valid action object with type PAGE_CLOSE."""
        action = cls._create_none_action()
        action.update(
            {
                "action_type": ActionTypes.PAGE_CLOSE,
            }
        )
        return action

    @classmethod
    @beartype
    def _create_click_action(cls,mark_id)->Action:
        action = cls._create_none_action()
        action.update(
            {
                "action_type":ActionTypes.CLICK,
                "mark_id":mark_id
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_hover_action(cls,mark_id:str)->Action:
        action = cls._create_none_action()
        action.update(
            {
                "action_type":ActionTypes.HOVER,
                "mark_id":mark_id
            }
        )
        return action
    
    @classmethod
    @beartype
    def _create_type_action(cls,text:str,mark_id:str):
        action = cls._create_none_action()
        action.update(
            {
                "action_type":ActionTypes.TYPE,
                "text":cls._keys2ids(text),
                "mark_id":mark_id
            }
        )
        return action
    
    @classmethod
    @beartype
    def action2str(
        cls,action:Action,action_set_tag:str="mark"
    )->str:
        """Return the string representation of an action

        Args:
            action (Action): the action
            action_set_tag (str): defalut to "mark"

        Returns:
            str: the string representation of the action
        """
        if action_set_tag == "mark":
            mark_id = action["mark_id"]
            match action["action_type"]:
                case ActionTypes.CLICK:
                    # [ID=X] xxxxx
                    action_str = f"click [{mark_id}]"
                case ActionTypes.TYPE:
                    action_str = f'type [{mark_id}] [{action["text"]}]'
                case ActionTypes.HOVER:
                    action_str = f"hover [{mark_id}]"
                case ActionTypes.SCROLL:
                    action_str = f"scroll [{action['direction']}]"
                case ActionTypes.KEY_PRESS:
                    action_str = f"press [{action['key_comb']}]"
                case ActionTypes.GOTO_URL:
                    action_str = f"goto [{action['url']}]"
                case ActionTypes.NEW_TAB:
                    action_str = "new_tab"
                case ActionTypes.PAGE_CLOSE:
                    action_str = "close_tab"
                case ActionTypes.GO_BACK:
                    action_str = "go_back"
                case ActionTypes.GO_FORWARD:
                    action_str = "go_forward"
                case ActionTypes.PAGE_FOCUS:
                    action_str = f"page_focus [{action['page_number']}]"
                case ActionTypes.STOP:
                    action_str = f"stop [{action['answer']}]"
                case ActionTypes.NONE:
                    action_str = "none"
                case _:
                    raise ValueError(
                        f"Unknown action type {action['action_type']}"
                    )
            return action_str
        else:
            raise NotImplementedError(f"Unknown action set tag {action_set_tag}")
    
    @classmethod
    @beartype
    def _execute_scroll(cls,direction: str, page: Page) -> None:
        if direction == "up":
            page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )
        elif direction == "left":
            page.evaluate(
                "(document.scrollingElement || document.body).scrollLeft = (document.scrollingElement || document.body).scrollLeft - window.innerWidth;"
            )
        elif direction == "right":
            page.evaluate(
                "(document.scrollingElement || document.body).scrollLeft = (document.scrollingElement || document.body).scrollLeft + window.innerWidth;"
            )
    
    @classmethod
    @beartype
    async def _aexecute_scroll(cls,direction: str, page: APage) -> None:
        if direction == "up":
            await page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop - window.innerHeight;"
            )
        elif direction == "down":
            await page.evaluate(
                "(document.scrollingElement || document.body).scrollTop = (document.scrollingElement || document.body).scrollTop + window.innerHeight;"
            )
        elif direction == "left":
            await page.evaluate(
                "(document.scrollingElement || document.body).scrollLeft = (document.scrollingElement || document.body).scrollLeft - window.innerWidth;"
            )
        elif direction == "right":
            await page.evaluate(
                "(document.scrollingElement || document.body).scrollLeft = (document.scrollingElement || document.body).scrollLeft + window.innerWidth;"
            )
    
    @classmethod
    @beartype
    def _execute_key_press(cls,key: str, page: Page) -> None:
        """
        Press a keyboard key.

        Args:
            key (str): The key to be pressed.
            page (Page): The page to perform the action on.

        Note:
            If the keyboard key contains "Meta" (usually referring to the Command key or Windows key) and the current system is not macOS,
            replace "Meta" with "Control", because on non-macOS systems, the "Control" key is typically used to perform functions similar to the "Command" key in macOS.
        """
        if "Meta" in key and "Mac" not in page.evaluate("navigator.platform"):
            key = key.replace("Meta", "Control")
        page.keyboard.press(key)
    
    @classmethod
    @beartype
    async def _aexecute_key_press(cls,key: str, page: APage) -> None:
        """
        Press a keyboard key.

        Args:
            key (str): The key to be pressed.
            page (Page): The page to perform the action on.

        Note:
            If the keyboard key contains "Meta" (usually referring to the Command key or Windows key) and the current system is not macOS,
            replace "Meta" with "Control", because on non-macOS systems, the "Control" key is typically used to perform functions similar to the "Command" key in macOS.
        """
        if "Meta" in key and "Mac" not in page.evaluate("navigator.platform"):
            key = key.replace("Meta", "Control")
        await page.keyboard.press(key)

    @classmethod
    @beartype
    def _execute_mouse_click(cls,left: float, top: float, page: Page) -> None:
        """Click at coordinates (left, top)."""
        viewport_size = page.viewport_size
        assert viewport_size
        page.mouse.click(
            left * viewport_size["width"], top * viewport_size["height"]
        )

    @classmethod
    @beartype
    async def _aexecute_mouse_click(cls,left: float, top: float, page: APage) -> None:
        """Click at coordinates (left, top)."""
        viewport_size = page.viewport_size
        assert viewport_size
        await page.mouse.click(
            left * viewport_size["width"], top * viewport_size["height"]
        )
    
    @classmethod
    @beartype
    def _execute_mouse_hover(cls,left: float, top: float, page: Page) -> None:
        """Click at coordinates (left, top)."""
        viewport_size = page.viewport_size
        assert viewport_size
        page.mouse.move(
            left * viewport_size["width"], top * viewport_size["height"]
        )
    
    @classmethod
    @beartype
    async def _aexecute_mouse_hover(cls,left: float, top: float, page: APage) -> None:
        """Click at coordinates (left, top)."""
        viewport_size = page.viewport_size
        assert viewport_size
        await page.mouse.move(
            left * viewport_size["width"], top * viewport_size["height"]
        )

    @classmethod
    @beartype
    def _execute_type(cls,keys: list[int], page: Page) -> None:
        """Send keystrokes to the focused element."""
        text = "".join([_id2key[key] for key in keys])
        page.keyboard.type(text)
    
    @classmethod
    @beartype
    async def _aexecute_type(cls,keys: list[int], page: APage) -> None:
        """Send keystrokes to the focused element."""
        text = "".join([_id2key[key] for key in keys])
        await page.keyboard.type(text)
    
    @classmethod
    @beartype
    def _execute_keyboard_type(cls,text: str, page: Page) -> None:
        """Fill the focused element with text."""
        page.keyboard.type(text)

    @classmethod
    @beartype
    async def _aexecute_keyboard_type(cls,text: str, page: APage) -> None:
        """Fill the focused element with text."""
        await page.keyboard.type(text)

    @classmethod
    @beartype
    def execute_action(
        cls,
        action:Action,
        page:Page,
        browser_ctx: BrowserContext,
        obseration_processor: ObservationProcessor,
        marked_elements:List[MarkedElement]
    )->Page:
        action_type = action["action_type"]
        match action_type:
            case ActionTypes.NONE:
                pass
            case ActionTypes.SCROLL:
                direction = ""
                if "up" in action["direction"]:
                    direction = "up"
                elif "down" in action["direction"]:
                    direction = "down"
                elif "left" in action["direction"]:
                    direction = "left"
                elif "right" in  action["direction"]:
                    direction = "right"
                
                cls._execute_scroll(direction,page)
                
            case ActionTypes.KEY_PRESS:
                keys = action["key_comb"]
                cls._execute_key_press(keys,page)
            
            case ActionTypes.CLICK:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    cls._execute_mouse_click(mark_center[0],mark_center[1],page)
                else:
                    raise NotImplementedError(
                    "No proper locator found for click action"
                )
            case ActionTypes.HOVER:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    cls._execute_mouse_hover(mark_center[0],mark_center[1],page)
                else:
                    raise NotImplementedError(
                    "No proper locator found for hover action"
                )

            case ActionTypes.TYPE:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    cls._execute_mouse_click(mark_center[0], mark_center[1], page)
                    cls._execute_type(action["text"], page)

            case ActionTypes.PAGE_FOCUS:
                page = browser_ctx.pages[action["page_number"]]
                page.bring_to_front()
            
            case ActionTypes.NEW_TAB:
                page = browser_ctx.new_page()
                page.client = page.context.new_cdp_session(page)
            
            case ActionTypes.GO_BACK:
                page.go_back()
            
            case ActionTypes.GO_FORWARD:
                page.go_forward()
            
            case ActionTypes.GOTO_URL:
                page.goto(action["url"])
            
            case ActionTypes.PAGE_CLOSE:
                page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page = browser_ctx.new_page()
            
            case _:
                raise ValueError(f"Unknown action type: {action_type}")
        return page
    
    @classmethod
    @beartype
    async def aexecute_action(
        cls,
        action:Action,
        page:APage,
        browser_ctx: BrowserContext,
        obseration_processor: ObservationProcessor,
        marked_elements:List[MarkedElement]
    )->APage:
        action_type = action["action_type"]
        match action_type:
            case ActionTypes.NONE:
                pass
            case ActionTypes.SCROLL:
                direction = ""
                if "up" in action["direction"]:
                    direction = "up"
                elif "down" in action["direction"]:
                    direction = "down"
                elif "left" in action["direction"]:
                    direction = "left"
                elif "right" in  action["direction"]:
                    direction = "right"
                
                await cls._aexecute_scroll(direction,page)
                
            case ActionTypes.KEY_PRESS:
                keys = action["key_comb"]
                await cls._aexecute_key_press(keys,page)
            
            case ActionTypes.CLICK:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    await cls._aexecute_mouse_click(mark_center[0],mark_center[1],page)
                else:
                    raise NotImplementedError(
                    "No proper locator found for click action"
                )
            case ActionTypes.HOVER:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    await cls._aexecute_mouse_hover(mark_center[0],mark_center[1],page)
                else:
                    raise NotImplementedError(
                    "No proper locator found for hover action"
                )

            case ActionTypes.TYPE:
                if action["mark_id"]:
                    mark_id = int(action["mark_id"])
                    mark_center = obseration_processor.get_marked_element_center(marked_elements[mark_id])
                    await cls._aexecute_mouse_click(mark_center[0], mark_center[1], page)
                    await cls._aexecute_type(action["text"], page)

            case ActionTypes.PAGE_FOCUS:
                page = browser_ctx.pages[action["page_number"]]
                await page.bring_to_front()
            
            case ActionTypes.NEW_TAB:
                page = browser_ctx.new_page()
                page.client =await page.context.new_cdp_session(page)
            
            case ActionTypes.GO_BACK:
                await page.go_back()
            
            case ActionTypes.GO_FORWARD:
                await page.go_forward()
            
            case ActionTypes.GOTO_URL:
                await page.goto(action["url"])
            
            case ActionTypes.PAGE_CLOSE:
                await page.close()
                if len(browser_ctx.pages) > 0:
                    page = browser_ctx.pages[-1]
                else:
                    page =await browser_ctx.new_page()
            
            case _:
                raise ValueError(f"Unknown action type: {action_type}")
        return page

    @classmethod
    @beartype
    def create_mark_based_action(cls,action_str:str)->Action:
        """Main function to return individual mark-based action"""
        action_str = action_str.strip()
        action_type = (
            action_str.split("[")[0].strip()
            if "[" in action_str
            else action_str.split()[0].strip()
        )
        match action_type:
            case "click":
                match = re.search(r"click ?\[(\d+)\]", action_str)
                if not match:
                    raise ValueError(f"Invalid click action {action_str}")
                mark_id = match.group(1)
                return cls._create_click_action(mark_id=mark_id)
            case "hover":
                match = re.search(r"hover ?\[(\d+)\]", action_str)
                if not match:
                    raise ValueError(f"Invalid hover action {action_str}")
                mark_id = match.group(1)
                return cls._create_hover_action(mark_id=mark_id)
            case "type":
                # add default enter flag
                if not (action_str.endswith("[0]") or action_str.endswith("[1]")):
                    action_str += " [1]"

                match = re.search(
                    r"type ?\[(\d+)\] ?\[(.+)\] ?\[(\d+)\]", action_str
                )
                if not match:
                    raise ValueError(f"Invalid type action {action_str}")
                mark_id, text, enter_flag = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                )
                if enter_flag == "1":
                    text += "\n"
                    print(664)
                return cls._create_type_action(text=text, mark_id=mark_id)
            case "press":
                match = re.search(r"press ?\[(.+)\]", action_str)
                if not match:
                    raise ValueError(f"Invalid press action {action_str}")
                key_comb = match.group(1)
                return cls._create_key_press_action(key_comb=key_comb)
            case "scroll":
                # up or down
                match = re.search(r"scroll ?\[?(up|down|left|right)\]?", action_str)
                if not match:
                    raise ValueError(f"Invalid scroll action {action_str}")
                direction = match.group(1)
                return cls._create_scroll_action(direction=direction)
            case "goto":
                match = re.search(r"goto ?\[(.+)\]", action_str)
                if not match:
                    raise ValueError(f"Invalid goto action {action_str}")
                url = match.group(1)
                return cls._create_goto_url_action(url=url)
            case "new_tab":
                return cls._create_new_tab_action()
            case "go_back":
                return cls._create_go_back_action()
            case "go_forward":
                return cls._create_go_forward_action()
            case "tab_focus":
                match = re.search(r"tab_focus ?\[(\d+)\]", action_str)
                if not match:
                    raise ValueError(
                        f"Invalid tab_focus action {action_str}"
                    )
                page_number = int(match.group(1))
                return cls._create_page_focus_action(page_number)
            
            case "close_tab":
                return cls._create_page_close_action()
            
            case "stop":  # stop answer
                match = re.search(r"stop ?\[(.+)\]", action_str)
                if not match:  # some tasks don't require an answer
                    answer = ""
                else:
                    answer = match.group(1)
                return cls._create_stop_action(answer)
        raise ValueError(f"Invalid action {action_str}")

        
        
    
    
    