from beartype import beartype
from .processor import (
    ImageObservationProcessor,
    ScreenshotType,
    ObservationHandler
)
from .utils import(
    numpy_to_png_bytes,
    Observation,
    MarkedElement
)
from .action import(
    Action,
    ActionHandler
)
from beartype.typing import(
    Any,
)
from playwright.sync_api import (
    Playwright, 
    sync_playwright, 
    expect,
    ViewportSize
    )




class WebEnvConfig:
    """Config of the website.
    """
    start_url: str

    def __init__(self,
        start_url: str = None         
        ) -> None:
        self.start_url = start_url
        pass

    @beartype
    def get(self,attr:str)->Any:
        """Get the value of the specified attribute."""
        return getattr(self, attr)

    def load_from_json(self)->None:
        
        pass
    pass



class BrowserEnv:


    @beartype
    def __init__(self,
        headless: bool = True,
        slow_mo: int = 0,
        observation_type:str = "image",
        image_observation_type: str ="mark",
        viewport_size: ViewportSize = {"width":1280,"height":720}
    ) -> None:
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_size = viewport_size
        self.if_runnning = False
        self.marked_elements = []
        match observation_type:
            case "image":
                self.image_observation_type = observation_type
                self.image_observation_type = image_observation_type
            case _:
                raise ValueError(
                    f"Unsupported observation type: {observation_type}"
                )
        self.observation_handler = ObservationHandler(
            image_observation_type=self.image_observation_type,
            viewport_size=self.viewport_size
        )
        pass
    
    @beartype
    def setup(self,web_env_config:WebEnvConfig):
        self.start_url = web_env_config.get("start_url")
        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.__enter__()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )

        self.context = self.browser.new_context(
            screen = self.viewport_size
        )

        pass
    

    def run(self)->None:
        if self.start_url:
            # single start_url
            self.page = self.context.new_page()
            self.client = self.page.context.new_cdp_session(self.page)
            self.page.goto(url = self.start_url)
        else:
            self.page = self.context.new_page()
            self.client  = self.page.context.new_cdp_session(self.page)
        self.if_runnning = True
        # get marked_elements in initial page
        self.marked_elements = self.get_observation()["marks"]


    def close(self)->None:
        if self.if_runnning == False:
            raise RuntimeError("The Web Environment is not running!")
        self.context.close()
        self.browser.close()
        self.context_manager.__exit__()
        self.if_runnning = False


    @beartype
    def get_observation(self)->dict[str,Observation]:
        obs_dict = self.observation_handler.get_observation(
            self.page,
            self.client
        )
        return obs_dict
    
    @beartype
    def execute(self,action:Action)->dict[str,Observation]:
        if self.if_runnning == False:
            raise RuntimeError("The Web Environment is not running!")
        
        # execute action for page
        self.page =  ActionHandler.execute_action(
            action=action,
            page = self.page,
            browser_ctx=self.context,
            obseration_processor=self.observation_handler.image_processor,
            marked_elements= self.marked_elements
        )

        obs_dict = self.get_observation()
        
        #update marked_elements
        self.marked_elements = obs_dict["marks"]

        return obs_dict
    
    def get_current_url(self)->str:
        assert self.if_runnning == True
        return self.page.url

    def get_all_urls(self):
        assert self.if_runnning == True
        #TODO
        pass