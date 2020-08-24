import numpy as np
from PIL import ImageGrab
import cv2
import pyautogui as pag
import time
import webbrowser

class Env:
    def __init__(self):
        print("LET\'S START!")
        self.done_mem = True
        self.episode = -1
        self.refresh = 10
        self.screen = None
        # Open T-Rex Game.
        webbrowser.open("http://www.trex-game.skipser.com/")
        time.sleep(3)
        # Full Screen mode
        pag.press('f11')
        time.sleep(1)

    def reset(self):
        self.episode += 1
        self.done_mem=True
        if self.episode%self.refresh == 0:
            pag.press('f5')
            time.sleep(1)

        pag.press('space')
        self.screen = self.capture()
        state = np.dstack((self.screen, self.screen))
        return state

    def is_done(self):
        screen = ImageGrab.grab().convert('L')
        screen = np.array(screen)
        # Check the Bottom of Screen
        done = (screen[len(screen)//2:,:]<245).any()
        if self.done_mem == False and done == True:
            return True
        else:
            self.done_mem = done
            return False

    def step(self, action):
        key = 'up' if action == 1 else 'down'
        pag.press(key)

        done = self.is_done()
        _screen = self.capture()
        state = np.dstack((self.screen, _screen))
        self.screen = _screen
        reward = -1 if done else 0.1
        return state, reward, done
    
    def capture(self):
        s = time.time()
        screen = ImageGrab.grab().convert('L')
        screen = np.array(screen, dtype=float)
        shape = screen.shape
        # Optimized 1920*1080
        screen = screen[shape[0]//6:shape[0]//3, shape[1]//3:-shape[1]//3]
        screen = cv2.resize(screen, dsize=(128, 64))
        screen = 255-screen
        screen = screen/255.
        e = time.time()
        return screen

    def close(self):
        pag.press('f11')
        pag.hotkey('ctrl', 'w')

    def alt_tap(self):
        pag.keyDown('alt'); pag.press('tab') ; pag.keyUp('alt')
