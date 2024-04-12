import numpy as np
import base64
import numpy.typing as npt
from PIL import Image
from io import BytesIO
from beartype.typing import (
    List,
    Tuple,
)
from beartype import beartype

def png_bytes_to_numpy(png: bytes) -> npt.NDArray[np.uint8]:
    """Convert png bytes to numpy array

    Example:

    >>> fig = go.Figure(go.Scatter(x=[1], y=[1]))
    >>> plt.imshow(png_bytes_to_numpy(fig.to_image('png')))
    """
    return np.array(Image.open(BytesIO(png)))

def numpy_to_png_bytes(numpy_array: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes.

    Args:
        numpy_array (np.ndarray): NumPy array representing an image.

    Returns:
        bytes: PNG bytes representing the image.
    """
    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(numpy_array)

    # Write image to bytes buffer as PNG
    png_bytes = BytesIO()
    pil_image.save(png_bytes, format='PNG')
    png_bytes.seek(0)

    return png_bytes.getvalue()

def png_bytes_to_base64(png: bytes) -> str:
    """
    Convert PNG image bytes to Base64 encoded string

    Args:
    png (bytes): Bytes representation of the PNG image

    Returns:
    str: Base64 encoded string
    """
    base64_encoded = base64.b64encode(png)
    base64_string = base64_encoded.decode('utf-8')
    return base64_string

class MarkedElement:
    """
    Represents a marked element with attributes such as index, text, tag, position, and size.

    Attributes:
        index (int): The index of the marked element.
        text (str): The text associated with the marked element.
        tag (str): The tag of the marked element.
        left (float): The left position of the marked element.
        top (float): The top position of the marked element.
        width (float): The width of the marked element.
        height (float): The height of the marked element.
    """

    @beartype
    def __init__(self, index: int, text: str, tag: str, left: float, top: float, width: float, height: float):
        """
        Initializes a new MarkedElement object.

        Args:
            index (int): The index of the marked element.
            text (str): The text associated with the marked element.
            tag (str): The tag of the marked element.
            left (float): The left position of the marked element.
            top (float): The top position of the marked element.
            width (float): The width of the marked element.
            height (float): The height of the marked element.
        """
        self.index = index
        self.text = text
        self.tag = tag
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a MarkedElement object from a dictionary.

        Args:
            data (dict): A dictionary containing attributes of the marked element.

        Returns:
            MarkedElement: A new MarkedElement object initialized with data from the dictionary.
        """
        return cls(
            index=data.get('index', 0),
            text=data.get('text', ''),
            tag=data.get('tag', ''),
            left=float(data.get('left', 0.0)),
            top=float(data.get('top', 0.0)),
            width=float(data.get('width', 0.0)),
            height=float(data.get('height', 0.0))
        )
    def __str__(self):
        return f"(index={self.index}, text='{self.text}', tag='{self.tag}')"

# text | image | (marks,image)
Observation = str | bytes | Tuple[List[MarkedElement],bytes]