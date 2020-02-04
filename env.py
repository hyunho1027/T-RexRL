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
        self.epoch = -1
        # Open T-Rex Game.
        webbrowser.open("http://www.trex-game.skipser.com/")
        time.sleep(3)
        # Full Screen mode
        pag.press('f11')
        time.sleep(1)

    def reset(self):
        self.epoch += 1
        self.done_mem=True
        pag.press('enter')
        state = np.dstack((self.capture(),self.capture()))
        return state

    def isDone(self):
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
        pag.keyDown(key)
        time.sleep(0.1)
        pag.keyUp(key)
    
        done = self.isDone()
        state = np.dstack((self.capture(),self.capture()))
        reward = -100 if done else 1
        return state, reward, done
    
    def capture(self):
        s = time.time()
        screen = ImageGrab.grab().convert('L')
        screen = np.array(screen, dtype=float)
        screen = screen[screen.shape[0]//6:screen.shape[0]//3,screen.shape[1]//3:-screen.shape[1]//3]
        screen = cv2.resize(screen, dsize=(128, 64))
        screen = 255 - screen
        # cv2.imwrite(f'./screen/img_{self.epoch}_{e-s}.png', screen)
        screen = (screen - 128)/128
        e = time.time()
        return screen

    def close(self):
        pag.press('f11')
        pag.hotkey('ctrl', 'w')
