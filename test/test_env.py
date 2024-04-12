import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
from browser_env.env import(
    WebEnvConfig,
    BrowserEnv
)
from browser_env.action import(
    ActionHandler
)
from browser_env.utils import(
    numpy_to_png_bytes
)
from browser_env.constant import(
    SPECIAL_KEYS
)
from browser_env.processor import ImageObservationProcessor,ScreenshotType
from browser_env.utils import numpy_to_png_bytes

import unittest

class TestBrowserEnv(unittest.TestCase):
    """Some random actions may be generated when testing the environment,
    so make sure that the test URLs are safe and reliable.
    """
    def setUp(self) -> None:
        self.start_url = "https://www.bing.com"
        self.web_env_config = WebEnvConfig(start_url=self.start_url)
        self.env = BrowserEnv(
            headless=False,
            slow_mo=1000)
        self.env.setup(self.web_env_config)
        return super().setUp()
    
    def tearDown(self) -> None:
        self.env.close()
        return super().tearDown()
        

    @unittest.skip('')
    def test_action_scroll(self):
        """test action: scroll
        """
        self.env.run()
        
        #down
        action_str = "scroll[down]"
        scroll_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(scroll_action)
        
        #up
        action_str = "scroll[up]"
        scroll_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(scroll_action)

        #left
        action_str = "scroll[left]"
        scroll_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(scroll_action)

        #right
        action_str = "scroll[right]"
        scroll_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(scroll_action)
        
        pass
    
    @unittest.skip('')
    def test_action_hover(self):
        self.env.run()

        #random hover
        marked_elements = self.env.marked_elements
        random_id = random.randint(0,len(marked_elements)-1)
        action_str = f"hover[{random_id}]"
        hover_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(hover_action)


    @unittest.skip('')
    def test_action_click(self):
        #test action click
        self.env.run()

        #random click
        marked_elements = self.env.marked_elements
        random_id = random.randint(0,len(marked_elements)-1)
        action_str = f"click[{random_id}]"
        click_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(click_action)


    @unittest.skip('')
    def test_action_press(self):
        #test action: press
        self.env.run()

        #random key comb
        key_comb = '+'.join(random.choice(SPECIAL_KEYS) for _ in range(random.randint(1, len(SPECIAL_KEYS))))
        action_str = f"press[{key_comb}]"
        press_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(press_action)

    #@unittest.skip('')
    def test_action_type(self):
        #test action: type
        #use google search
        self.env.start_url = 'https://google.com/'
        self.env.run()
        action_str = 'type[4][current president of the united states][1]'
        type_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(type_action)
        
    @unittest.skip('')
    def test_action_others(self):
        #test action: new_tab, goto url, go_back, go_forward, tab_focuse, close_tab
        self.env.run()
        #open new tab
        action_str = 'new_tab'
        new_tab_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(new_tab_action)

        #goto url
        action_str = 'goto[https://www.google.com]'
        goto_url_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(goto_url_action)

        #go_back
        action_str = 'go_back'
        go_back_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(go_back_action)

        #go_forward
        action_str = 'go_forward'
        go_forward_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(go_forward_action)

        #tab_focus
        action_str = 'tab_focus[0]'
        tab_focus_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(tab_focus_action)

        #close_tab
        action_str = 'close_tab'
        close_tab_action = ActionHandler.create_mark_based_action(action_str)
        self.env.execute(close_tab_action)

    

if __name__ == '__main__':
    unittest.main()