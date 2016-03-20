# TensorFlow On AWS with GPU support

The following are notes for setting up an AWS instance with GPU support for TensorFlow.  This is accurate as of 3/19/2015, and things seem to change rapidly so YMMV.  This material was cribbed from the following sources which may be helpful:

* [https://gist.github.com/erikbern/78ba519b97b440e10640](https://gist.github.com/erikbern/78ba519b97b440e10640)
* [https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN)](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN))
* [http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/](http://tleyden.github.io/blog/2014/10/25/cuda-6-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/)

### Get an AWS GPU instance

Here is what I used: `g2.2xlarge` with ubuntu AMI `ubuntu-trusty-14.04-amd64-server-20160114.5 (ami-06116566)`.  During instance setup I increased the root partition from 8gb to 16gb.  You will need this to build (or you can build on the /mnt ebs partition).

`ssh` in and get ready to rumble!

### Get the basics

    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get install -y build-essential python-pip python-dev git python-numpy swig python-dev default-jdk zip zlib1g-dev

    sudo apt-get install -y linux-image-extra-virtual
    sudo reboot
    sudo apt-get install -y linux-source linux-headers-`uname -r`

### Get CUDA

google "cuda download" and find the link.  You'll want linux, x86_64, ubuntu, 14.04 if you are using the same ubuntu AMI I am.   You want the "run local" version.  Right now the url is [http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run)

    wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run 
    chmod +x cuda_7.5.18_linux.run
    ./cuda_7.5.18_linux.run -extract=`pwd`/nvidia_installers
    cd nvidia_installers
    sudo ./NVIDIA-Linux-x86_64-342.39.run 

The driver will likely claim there is a conflict with nouveau, and offer to write a file to fix it.  Let it.  It will then tell you to reboot and the installer will exit.  Before rebooting update the init ramdisk:

    sudo update-initramfs -u
    sudo reboot

Now you can run the installer again.  I used defaults for all questions.  If you still get grief about nouvea try googling or refer to [this](https://gist.github.com/erikbern/78ba519b97b440e10640).  Note that during one dry run I was told to uninstall the existing NVIDIA driver, and was given a command do so. I did it and it worked, so if that comes up just follow along.

    cd nvdia_installers
    sudo ./NVIDIA-Linux-x86_64-342.39.run 
    sudo modprobe nvidia

Now that the NVIDIA drivers are installed, install cuda.  I accepted all defaults, including having it install in `/usr/local/cuda-7.5` and creating an alias for `/usr/local/cuda`.

    sudo ./cuda-linux64-rel-7.0.28-19326674.run
    cd

Add cuda to .bashrc

    echo “export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64” >> ~/.bashrc
    echo “export CUDA_HOME=/usr/local/cuda” >> ~/.bashrc
    . ~/.bashrc

### Install CUDNN

Install the Cuda deep neural network library.  Download it from the nvidia site.  As of this writing it requires you to register  as a developer and fill out a survey.  Play along.  You will end up with a file like `cudnn-7.0-linux-x64-v4.0-prod.tgz `.  You want to install it over the `/usr/local/cuda` cuda installation.

    cd /usr/local
    sudo tar xzvf ~/cudnn-7.0-linux-x64-v4.0-prod.tgz 

### Install bazel

In order to build TensorFlow with "older" GPU support we need the Google opensource build tool bazel.  To install bazel we need java 8.  So first check what version of java you have by running `java -version`.  Most lilely its 1.7x.  To install java 8:

    add-apt-repository ppa:webupd8team/java
    sudo apt-get update
    sudo apt-get install oracle-java8-installer

Answer the installer questions with the defaults.  I make no warranties as to what part of your soul you are selling to Oracle by agreeing to the TOS.  Check again `java -version`.  Should be 1.8x.  Define JAVA_HOME in your .bashrc:

    echo “export JAVA_HOME=/usr/lib/jvm/java-8-oracle” >> ~/.bashrc
    . ~/.bashrc

Now on to bazel.  As of this writing we need bazel 0.1.4 exactly, which is a bit stale.  As of your reading, maybe its different!  Google it if this doesn't work.

    cd
    wget https://github.com/bazelbuild/bazel/releases/download/0.1.4/bazel-0.1.4-installer-linux-x86_64.sh
    chmod +x bazel-0.1.4-installer-linux-x86_64.sh 
    sudo ./bazel-0.1.4-installer-linux-x86_64.sh 
    
### Build TensorFlow

    git clone --recurse-submodules https://github.com/tensorflow/tensorflow
    cd tensorflow

The key thing to building TensorFlow is that we need to build with GPU support, and we need to build for "compute capability for 3.0", which is slightly older and is compatible with the generation of GPUs AWS currently has.  If you google there are lots of people asking for this, so support was unofficially added.  So configure as follows:

    TF_UNOFFICIAL_SETTING=1 ./configure

All of the questions can be answered with default except these two:

    Do you wish to build TensorFlow with GPU support? [y/N] y

    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size.
    [Default is: "3.5,5.2"]: 3.0

Now build and install (this will take 10-20 minutes)

    bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
    bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_pkg
    sudo pip install ~/tensorflow_pkg/tensorflow-0.7.1-cp27-none-linux_x86_64.whl

### Test with the MNIST tutorial

Use the mnist.py that is in this repo.  It is the moral equivalent of this [tensorflow tutorial](https://www.tensorflow.org/versions/r0.7/tutorials/mnist/pros/index.html).

    cd
    git clone https://github.com/gtoubassi/TensorFlowOnAWSwithGPU.git
    cd TensorFlowOnAWSwithGPU.git
    python mnist.py
    

It should spew 10 lines or so of boilerplate, download the MNIST dataset, and then start spitting out lines like this:

    step 0, training accuracy 0.04
    step 100, training accuracy 0.86
    step 200, training accuracy 0.9
    step 300, training accuracy 0.86

The lines should print every 1-2 seconds.  If you give it 5 minutes to train (it needs to get to step 20000) it will print out that it eached 99.2% accurace.  Pretty awesome for a <100 line script!  To make sure you are getting the full power of the GPU, try running the script on the cpu (not gpu):

    python mnist.py --force-cpu
    
Although it will spew the same business about firing up the GPU the actual computations are instructed to be on the cpu, so you will see each batch printed out every 15 seconds instead of 1.5 seconds.  So about 10x faster (kinda surprised its not even more).
 

