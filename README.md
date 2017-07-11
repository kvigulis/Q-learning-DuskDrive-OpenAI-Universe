# Q-learning-DuskDrive-OpenAI-Universe


Easy to set up...

Dependencies: 

<t>Set up tesorflow or tensorflow-gpu:<br>
For help refer to this slightly outdated resource on how to set up CUDA and cuDNN (to apply for newer versions just correct some paths with '../local/CUDA-8.0/..', for example...) :
http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/<br>
Remember to download cuDNN 5.1 not 6.0<br><br>
Of course, OpenAI Universe. First install gym with ```sudo pip install -e '.[all]'``` while in the gym directory.<br><br>
Other python dependencies:<br>
```sudo pip install numpy```<br>
```sudo pip install Pillow```<br>
```sudo pip install scipy```<br>

Run RL.py to start training the agent. TensorFlow checkpoint saved every 10 runs of the game by default.<br><br>


<i>This attempt was inspired from a tutorial by Hvass-Labs who's author is Magnus Erik Hvass Pedersen.

This is a very plain implementation with just the minimum of code required to run the Q-Learning Algorithm.
