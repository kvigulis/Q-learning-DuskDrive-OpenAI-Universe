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
```sudo pip install scipy```<br>
...or just set up with Anaconda<br><br>
 * (For PyCharm users) To use TensorFlow in PyCharm IDE edit project configuration and add ```LD_LIBRARY_PATH``` with ```'/usr/local/cuda-8.0/lib64'``` to the environment variables.<br><br>


Run RL.py to start training the agent. TensorFlow checkpoint saved every 10 runs of the game by default.<br><br>


<i>This attempt was inspired from a tutorial by Hvass-Labs who's author is Magnus Erik Hvass Pedersen.

This is a very plain implementation with just the minimum of code required to run the Q-Learning Algorithm.


</i>Reminder for me on fresh Ubuntu installation:
* Install nvidia drivers after: <br>
```sudo add-apt-repository ppa:graphics-drivers/ppa```<br>
```sudo apt update```<br>
and then go to 'Software & Updates', 'Additional Drivers' and choose a driver.
* Download cuda-8.0 linux_86_64x.deb and cudnn5.1.deb and install both... restart your PC for them to work.
* Install Anaconda x86_64.sh, follow the Tensorflow tutorial and use: https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl
to install the tensorflow-gpu version in the conda environment. The miniconda3 will belong to 'root', so change it to your user before installing tensorflow. Otherwise permission to write will be denied.<br>
