import pyautogui as pag
import cv2 
import numpy as np
img = pag.screenshot

img = np.array(img)

cv2.imshow('img', img[:len(img)//2,len(img[0])//4:-len(img[0])//4])
cv2.waitKey(0)
cv2.destroyAllWindows()