# Q-learning-DuskDrive-OpenAI-Universe


Very easy to set up...

Dependencies: 

Set up tesorflow or tensorflow-gpu.<br>
Slightly outdated resource on how to set up CUDA and cuDNN (mainly just correct some paths with 'CUDA-8.0'):
http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/<br>
Of course, OpenAI Universe. First install gym with sudo pip install -e '.[all]'.<br>
Other python dependencies:<br>
pip install numpy<br>
pip install Pillow<br>
pip install scipy<br>

Run RL.py to start training the agent. TensorFlow checkpoint saved every 10 runs of the game by default.<br><br>


This attempt was inspired from a tutorial by Hvass-Labs who's author is Magnus Erik Hvass Pedersen.

This is a very plain implementation with just the minimum of code required to run the Q-Learning Algorithm.
