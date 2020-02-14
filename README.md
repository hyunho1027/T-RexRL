# T-RexRL

<p align="center">
    <img src="./resrc/T-Rex.gif" width=70%>
</p>

# Introduction
This is [T-Rex Runner](http://www.trex-game.skipser.com/)

## Agent
The agent is T-Rex

## Observation
This environment provides only visual observation

### Visual Observation
<img src="./resrc/capture_screen.png" width=256>

shape: (64, 128, 2)
(* stack 2 capture screens)

## Action
Jump(1) or Duck(0)

## Reward
-1 if terminal else 0.1

# Installation
```cmd
git clone https://github.com/hyunho1027/T-RexRL
```

## Requirements
- Python 3.7
- image
- pyautogui
- Tensorflow 2.0
- cv2
- webbrowser

## Usage
```cmd
python main.py
```
