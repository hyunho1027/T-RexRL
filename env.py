import numpy as np
import pyautogui as pag
import time
import webbrowser

class Env:
    def __init__(self):
        print("LET\'S START!")
        self.episode = -1
        self.refresh = 10
        self.remember = None
        self.size = pag.screenshot().size
        self.target_size = (128,64)

        # Optimized 1920*1080
        self.state_region = (self.size[0]//3, self.size[1]//6,
                             self.size[0]//3, self.size[1]//6)
        self.done_region = ((self.size[0]//10)*9, (self.size[1]//10)*9,
                             self.size[0]//10   ,  self.size[1]//10)

        # Open T-Rex Game.
        webbrowser.open("http://www.trex-game.skipser.com/")
        time.sleep(3)

        # Full Screen mode
        pag.press('f11')
        time.sleep(1)

    def reset(self):
        self.episode += 1
        if self.episode%self.refresh == 0:
            pag.press('f5')
            time.sleep(1)

        is_start = False
        while not is_start:
            pag.press('up')
            screen = pag.screenshot(region=self.done_region).convert('L')
            screen = np.array(screen)
            # Check the Bottom of Screen
            is_start = (screen>128).all()
        pag.press('up')

        screen = self.capture()
        state = np.dstack((screen, screen))
        self.remember = screen
        return state

    def is_done(self):
        screen = pag.screenshot(region=self.done_region).convert('L')
        screen = np.array(screen)
        # Check the Bottom of Screen
        done = (screen<128).any()
        return done

    def step(self, action):
        key = 'up' if action == 1 else 'down'
        pag.press(key)

        screen = self.capture()
        state = np.dstack((self.remember, screen))
        self.remember = screen
        done = self.is_done()
        reward = -1 if done else 0.1
        return state, reward, done
    
    def capture(self):
        screen = pag.screenshot(region=self.state_region).convert('L').resize(self.target_size)
        screen = np.array(screen, dtype=float)
        screen = (255-screen)/255.
        return screen

    def close(self):
        pag.press('f11')
        pag.hotkey('ctrl', 'w')

    def alt_tap(self):
        pag.keyDown('alt'); pag.press('tab') ; pag.keyUp('alt')
