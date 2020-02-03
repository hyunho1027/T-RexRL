import wx
import numpy as np

app = wx.App()
screen= wx.ScreenDC()
bmp = wx.Bitmap(1920,1080)

mem = wx.MemoryDC(bmp)
mem.Blit(0,0,1920,1080,screen,0,0)
del mem

print(np.array(bmp))