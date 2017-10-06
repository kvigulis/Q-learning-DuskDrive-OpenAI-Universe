# Q-learning-DuskDrive-OpenAI-Universe

Easy to set up...

Set up tesorflow or tensorflow-gpu:<br>
____

For setting up ```tensorflow-gpu``` a on fresh Ubuntu installation:
* Install nvidia drivers after: <br>
```sudo apt-add-repository ppa:graphics-drivers/ppa```<br>
```sudo apt-get update```<br>
and then go to 'Software & Updates', 'Additional Drivers' and choose a driver.
* Download cuda-8.0 linux_86_64x.deb and cudnn5.1.deb and install both... restart your PC for them to work.
* Install Anaconda x86_64.sh, follow the Tensorflow tutorial and use: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl
to install the tensorflow-gpu version in the conda environment. 
___

Other dependecies:

* Of course, OpenAI Universe: https://github.com/openai/universe
* Install go-vncdiver with OpenGL support. Look here: https://github.com/openai/go-vncdriver

* (Ignore if you have Anaconda) Python dependencies :<br>
```sudo pip install numpy```<br>
```sudo pip install Pillow```<br>
```sudo pip install scipy```<br>


* The python file must be run as root. For running applications as root from UnityLaucher (Ubuntu's bar on the left) follow this: https://askubuntu.com/questions/118822/how-to-launch-application-as-root-from-unity-launcher
 
* (Might need for PyCharm users) To use TensorFlow in PyCharm IDE edit project configuration and add ```LD_LIBRARY_PATH``` with ```'/usr/local/cuda-8.0/lib64'``` to the environment variables.<br><br>


Run RL.py to start training the agent. TensorFlow checkpoint saved every 10 runs of the game by default.<br><br>


<i>This attempt was inspired from a tutorial by Hvass-Labs who's author is Magnus Erik Hvass Pedersen.
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/16_Reinforcement_Learning.ipynb

This is a very plain implementation with just the minimum of code required to run the Q-Learning Algorithm.

___

</i>
