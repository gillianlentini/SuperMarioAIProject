'''
   This code was written by following the following PyTorch tutorial
   Tutorial Title: TRAIN A MARIO-PLAYING RL AGENT
   Project Title: MadMario
   Author: Yuansong Feng, Suraj Subramanian, Howard Wang, Steven Guo
   Date: June 2020
   Code version: 2.0
   Availability: https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html#train-a-mario-playing-rl-agent
                 https://github.com/YuansongFeng/MadMario
'''

from IPython.display import HTML
from IPython import display as ipythondisplay
import glob
import io
import base64

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""


def show_video():
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
