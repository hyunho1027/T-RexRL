import numpy as np
from PIL import ImageGrab
import cv2

grab_img = ImageGrab.grab().convert('L')
img = np.array(grab_img)
img = img[img.shape[0]//6:img.shape[0]//3,len(img[0])//3:-len(img[0])//3]
img = cv2.resize(img, dsize=(128, 64))
img = 255 - img
print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('img.png', img)
