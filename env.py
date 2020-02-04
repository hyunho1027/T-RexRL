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
        self.num = 0
        self.epoch = -1
        # Open T-Rex Game.
        webbrowser.open("http://www.trex-game.skipser.com/")
        time.sleep(1)
        # Full Screen mode
        pag.press('f11')
        time.sleep(3)

    def reset(self):
        self.epoch += 1
        self.num = 0
        self.done_mem=True
        pag.press('enter')
        state = np.reshape(np.r_[self.capture(),self.capture()], (128,128,1))
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
        state = np.reshape(np.r_[self.capture(),self.capture()], (128,128,1))
        reward = -1 if done else 0.1
        return state, reward, done
    
    def capture(self):
        s = time.time()
        screen = ImageGrab.grab().convert('L')
        screen = np.array(screen, dtype=float)
        screen = screen[screen.shape[0]//6:screen.shape[0]//3,screen.shape[1]//3:-screen.shape[1]//3]
        screen = cv2.resize(screen, dsize=(128, 64))
        screen = 255 - screen
        e = time.time()
        # cv2.imwrite(f'./screen/img_{self.epoch}_{self.num}_{e-s}.png', screen)
        self.num +=1
        return screen

    def close(self):
        pag.hotkey('ctrl', 'w')
