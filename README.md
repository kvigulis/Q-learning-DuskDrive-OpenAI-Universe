# Q-learning-DuskDrive-OpenAI-Universe


Easy to set up...

#### Dependencies: 

* Set up tesorflow or tensorflow-gpu:<br>
For help refer to this slightly outdated resource on how to set up CUDA and cuDNN (to apply for newer versions just     correct some paths with '../local/CUDA-8.0/..', for example...) :
http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/<br>
Remember to download cuDNN 5.1 not 6.0<br><br>
* Of course, OpenAI Universe (recommended in conda environment). Set it up by following instructions here: https://alliseesolutions.wordpress.com/2016/12/08/openai-universe-installation-guide-ubuntu-16-04/First <br><br>
* Other python dependencies:<br>
```sudo pip install numpy```<br>
```sudo pip install Pillow```<br>
```sudo pip install scipy```<br><br>
 * To use TensorFlow in PyCharm IDE edit project configuration and add ```LD_LIBRARY_PATH``` with ```'/usr/local/cuda-8.0/lib64'``` to the environment variables.<br><br>


Run RL.py to start training the agent. TensorFlow checkpoint saved every 10 runs of the game by default.<br><br>


<i>This attempt was inspired from a tutorial by Hvass-Labs who's author is Magnus Erik Hvass Pedersen.

This is a very plain implementation with just the minimum of code required to run the Q-Learning Algorithm.
