#TODO add TextObservationProcessor


from abc import( 
    ABC,
    abstractmethod
)
from enum import IntEnum
import numpy as np
import numpy.typing as npt
from .utils import(
    MarkedElement,
    Observation,
    png_bytes_to_numpy
)
from beartype import beartype


from playwright.sync_api import CDPSession, Page, ViewportSize
from .mark_logic import js_marking


class ScreenshotType(IntEnum):
    MARK = 0
    UNMARK = 1
    def __str__(self) -> str:
        return f"ScreenshotType.{self.name}"


class ObservationProcessor(ABC):

    @abstractmethod
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError




class ImageObservationProcessor(ObservationProcessor):

    @beartype
    def __init__(
        self, 
        observation_type: str, 
        screenshot_type: ScreenshotType, 
        viewport_size: ViewportSize,
    ):
        """Initialize ImageObservationProcessor.

        Args:
            observation_type (str): A string describing the type of observation being made.
            screenshot_type (ScreenshotType): An enum indicating whether to capture a "mark" or "unmark" screenshot.
        """
        self.observation_type = observation_type
        self.screenshot_type = screenshot_type
        self.observation_tag = "image"
        self.viewport_size = viewport_size

    @beartype
    def process(self, page: Page, client: CDPSession) -> Observation:
        """Process the web page and return the screenshot as a NumPy array.

        Args:
            page (Page): The web page to process.
            client (CDPSession): The CDP session for interacting with the page.

        Returns:
        Tuple[List[MarkedElement], npt.NDArray[np.uint8]]] / npt.NDArray[np.uint8]] / None: 
            - If ScreenshotType is MARK, returns a tuple containing a list of MarkedElement objects representing the marked elements and a NumPy array representing the screenshot of the web page.
            - If ScreenshotType is UNMARK, returns only the screenshot as a NumPy array.
            - If an exception occurs, returns None.
        """
        try:
            if self.screenshot_type == ScreenshotType.MARK:
                marks=page.eval_on_selector(selector="body",expression=js_marking)
                marked_elements = [MarkedElement.from_dict(mark_data) for mark_data in marks]
                screenshot = page.screenshot()
                return marked_elements,screenshot
            elif self.screenshot_type == ScreenshotType.UNMARK:
                screenshot = page.screenshot()
                return screenshot
        except:
            page.wait_for_load_state("load")
            if self.screenshot_type == ScreenshotType.MARK:
                marks=page.eval_on_selector(selector="body",expression=js_marking)
                marked_elements = [MarkedElement.from_dict(mark_data) for mark_data in marks]
                screenshot = png_bytes_to_numpy(page.screenshot())
                return marked_elements,screenshot
            elif self.screenshot_type == ScreenshotType.UNMARK:
                screenshot = png_bytes_to_numpy(page.screenshot())
                return screenshot
        return None
    
    @beartype
    def get_marked_element_center(self, marked_element:MarkedElement) -> tuple[float, float]:
        x,y,width,height = marked_element.left,marked_element.top,marked_element.width,marked_element.height
        center_x = x + width / 2
        center_y = y + height / 2
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )
    
    
class ObservationHandler:
    """
    Class to handle observations.

    Attributes:
        image_processor: Instance of ImageObservationProcessor to process image observations.
    """

    @beartype
    def __init__(
        self,
        image_observation_type: str,
        viewport_size: ViewportSize,
    ) -> None:
        """
        Constructor for ObservationHandler.

        Args:
            image_observation_type (str): Type of image observation. Defaults to "unmark".
            screenshot_type (ScreenshotType): Type of screenshot. Defaults to ScreenshotType.UNMARK.
        """
        self.image_observation_type = image_observation_type
        self.viewport_size = viewport_size
        match self.image_observation_type:
            case "mark":
                self.image_processor = ImageObservationProcessor(
                    observation_type=image_observation_type, 
                    screenshot_type=ScreenshotType.MARK,
                    viewport_size=self.viewport_size
                )
            case "unmark":
                self.image_processor = ImageObservationProcessor(
                    observation_type=image_observation_type, 
                    screenshot_type=ScreenshotType.MARK,
                    viewport_size=self.viewport_size
                )
            case _:
                raise ValueError(f"unknow image_observation type: {image_observation_type}")
        

    @beartype
    def get_observation(self, page: Page, client: CDPSession) -> dict[str, Observation]:
        """
        Get observations from the page.

        Args:
            page (Page): Page object to get observations from.
            client (CDPSession): CDPSession object for communication with the browser.

        Returns:
            dict[str, Observation]: Dictionary containing observations.
        """
        if self.image_observation_type == "mark":
            marks,marked_image = self.image_processor.process(page, client)
            return {
                "image": marked_image,
                "marks": marks
            }
        
        elif self.image_observation_type == "unmark":
            unmarked_image = self.image_processor.process(page,client)
            return{
                "image":unmarked_image,
                "marks":[]
            }
        
        return None

