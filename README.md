# Installation notes for seting-up Tensorflow

## Install Python
Python is automatically installed on Ubuntu. Take a moment to confirm (by issuing a python -V command) that one of the following Python versions is already installed on your system:

Python 2.7
Python 3.3+
The pip or pip3 package manager is usually installed on Ubuntu. Take a moment to confirm (by issuing a pip -V or pip3 -V command) that pip or pip3 is installed. We strongly recommend version 8.1 or higher of pip or pip3. If Version 8.1 or later is not installed, issue the following command, which will either install or upgrade to the latest pip version:
```
$ sudo apt-get install python-pip python-dev   # for Python 2.7
$ sudo apt-get install python3-pip python3-dev # for Python 3.n
```

## Install Bazel
* Install JDK 8
Install JDK 8 by using:

```sudo apt-get install openjdk-8-jdk```

On Ubuntu 14.04 LTS you'll have to use a PPA:

```
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update && sudo apt-get install oracle-java8-installer
```

* Add Bazel distribution URI as a package source (one time setup)
```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```
If you want to install the testing version of Bazel, replace stable with testing.

* Install and update Bazel
```
sudo apt-get update && sudo apt-get install bazel
```
Once installed, you can upgrade to a newer version of Bazel with:
```
sudo apt-get upgrade bazel
```

## Install GPU support for TensorFlow
If you are building TensorFlow without GPU support, skip this section.

The following NVIDIA hardware must be installed on your system:

GPU card with CUDA Compute Capability 3.0 or higher. See NVIDIA documentation for a list of supported GPU cards.
The following NVIDIA software must be installed on your system:

NVIDIA's Cuda Toolkit (>= 7.0). We recommend version 8.0. For details, see NVIDIA's documentation. Ensure that you append the relevant Cuda pathnames to the `LD_LIBRARY_PATH` environment variable as described in the NVIDIA documentation.
The NVIDIA drivers associated with NVIDIA's Cuda Toolkit.
cuDNN (>= v3). We recommend version 5.1. For details, see NVIDIA's documentation, particularly the description of appending the appropriate pathname to your `LD_LIBRARY_PATH` environment variable.
Finally, you must also install libcupti-dev by invoking the following command:
```
 $ sudo apt-get install libcupti-dev 
```
### Install CUDA and cuDNN
[NVIDIA CUDA Installation Guide for Linux](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/#axzz4VZnqTJ2A)

[NVIDIA cuDNN](https://developer.nvidia.com/cudnn)



## Install Tensorflow
### Windows
TensorFlow supports Python 3.5.x and 3.6.x on Windows. Note that Python 3 comes with the pip3 package manager, which is the program you'll use to install TensorFlow.

To install TensorFlow, start a terminal. Then issue the appropriate pip3 install command in that terminal. To install the CPU-only version of TensorFlow, enter the following command:
```
C:\> pip3 install --upgrade tensorflow
```
To install the GPU version of TensorFlow, enter the following command:
```
C:\> pip3 install --upgrade tensorflow-gpu
```

### Ubuntu
Assuming the prerequisite software is installed on your Linux host, take the following steps:

Install TensorFlow by invoking one of the following commands:
```
$ pip install tensorflow      # Python 2.7; CPU support (no GPU support)
$ pip3 install tensorflow     # Python 3.n; CPU support (no GPU support)
$ pip install tensorflow-gpu  # Python 2.7;  GPU support
$ pip3 install tensorflow-gpu # Python 3.n; GPU support 
```
If the preceding command runs to completion, you should now validate your installation.
(Optional.) If Step 1 failed, install the latest version of TensorFlow by issuing a command of the following format:
```
$ sudo pip  install --upgrade tfBinaryURL   # Python 2.7
$ sudo pip3 install --upgrade tfBinaryURL   # Python 3.n
 ```
where tfBinaryURL identifies the URL of the TensorFlow Python package. The appropriate value of tfBinaryURL depends on the operating system, Python version, and GPU support. Find the appropriate value for tfBinaryURL here. For example, to install TensorFlow for Linux, Python 3.4, and CPU-only support, issue the following command:
```
$ sudo pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp34-cp34m-linux_x86_64.whl
```

### Compiling from the source
#### Clone the TensorFlow repository

Start the process of building TensorFlow by cloning a TensorFlow repository.

To clone the latest TensorFlow repository, issue the following command:
```
$ git clone https://github.com/tensorflow/tensorflow 
```
The preceding git clone command creates a subdirectory named tensorflow. After cloning, you may optionally build a specific branch (such as a release branch) by invoking the following commands:
```
$ cd tensorflow
```

#### Prepare environment for Linux

Before building TensorFlow on Linux, install the following build tools on your system:

  * bazel
  * TensorFlow Python dependencies
  * optionally, NVIDIA packages to support TensorFlow for GPU.

##### Install TensorFlow Python dependencies

To install TensorFlow, you must install the following packages:

numpy, which is a numerical processing package that TensorFlow requires.
dev, which enables adding extensions to Python.
pip, which enables you to install and manage certain Python packages.
wheel, which enables you to manage Python compressed packages in the wheel (.whl) format.
To install these packages for Python 2.7, issue the following command:
```
$ sudo apt-get install python-numpy python-dev python-pip python-wheel
```
To install these packages for Python 3.n, issue the following command:
```
$ sudo apt-get install python3-numpy python3-dev python3-pip python3-wheel
```

#### Configure the installation
The root of the source tree contains a bash script named configure. This script asks you to identify the pathname of all relevant TensorFlow dependencies and specify other build configuration options such as compiler flags. You must run this script prior to creating the pip package and installing TensorFlow.

If you wish to build TensorFlow with GPU, configure will ask you to specify the version numbers of Cuda and cuDNN. If several versions of Cuda or cuDNN are installed on your system, explicitly select the desired version instead of relying on the default.

One of the questions that configure will ask is as follows:
```
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]
```
This question refers to a later phase in which you'll use bazel to build the pip package. We recommend accepting the default (`-march=native`), which will optimize the generated code for your local machine's CPU type. However, if you are building TensorFlow on one CPU type but will run TensorFlow on a different CPU type, then consider specifying a more specific optimization flag as described in the gcc documentation.

An example as follows, change the items corresondingly (e.g. Python path):
```
$ cd tensorflow  # cd to the top-level directory created
$ ./configure
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python2.7
Found possible Python library paths:
  /usr/local/lib/python2.7/dist-packages
  /usr/lib/python2.7/dist-packages
Please input the desired Python library path to use.  Default is [/usr/lib/python2.7/dist-packages]

Using python library path: /usr/local/lib/python2.7/dist-packages
Do you wish to build TensorFlow with MKL support? [y/N] (N if using AMD Ryzen)
No MKL support will be enabled for TensorFlow
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]:
Do you wish to use jemalloc as the malloc implementation? [Y/n]
jemalloc enabled
Do you wish to build TensorFlow with Google Cloud Platform support? [y/N]
No Google Cloud Platform support will be enabled for TensorFlow
Do you wish to build TensorFlow with Hadoop File System support? [y/N]
No Hadoop File System support will be enabled for TensorFlow
Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N]
No XLA support will be enabled for TensorFlow
Do you wish to build TensorFlow with VERBS support? [y/N]
No VERBS support will be enabled for TensorFlow
Do you wish to build TensorFlow with OpenCL support? [y/N]
No OpenCL support will be enabled for TensorFlow
Do you wish to build TensorFlow with CUDA support? [y/N] Y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 8.0]: 8.0
Please specify the location where CUDA 8.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 6.0]: 6
Please specify the location where cuDNN 6 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.0
Do you wish to build TensorFlow with MPI support? [y/N] 
MPI support will not be enabled for TensorFlow
Configuration finished
```

If you told configure to build for GPU support, then configure will create a canonical set of symbolic links to the Cuda libraries on your system. Therefore, every time you change the Cuda library paths, you must rerun the configure script before re-invoking the `bazel build` command.

Note the following:

Although it is possible to build both Cuda and non-Cuda configs under the same source tree, we recommend running `bazel clean` when switching between these two configurations in the same source tree.
If you don't run the configure script before running the `bazel build` command, the bazel build command will fail

#### Build the pip package

To build a pip package for TensorFlow with CPU-only support, you would typically invoke the following command:
```
$ bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```
or following command with optimization flags explicitly defined (`--copt=-march=native` checks the detailed instruction configuration from the CPU):
```
bazel build -c opt --copt=-march=native --copt=-mfpmath=both --config=cuda -k //tensorflow/tools/pip_package:build_pip_package
```
or more explicitly (not recommended):
```
bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k //tensorflow/tools/pip_package:build_pip_package
```
Use `gcc -march=native -Q --help=target` to check the supported instructions

To build a pip package for TensorFlow with GPU support, invoke the following command (basic without optimization flags):
```
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_packag
```

NOTE on gcc 5 or later: the binary pip packages available on the TensorFlow website are built with gcc 4, which uses the older ABI. To make your build compatible with the older ABI, you need to add `--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"` to your `bazel build` command. ABI compatibility allows custom ops built against the TensorFlow pip package to continue to work against your built package.

Tip: By default, building TensorFlow from sources consumes a lot of RAM. If RAM is an issue on your system, you may limit RAM usage by specifying `--local_resources 2048,.5,1.0` while invoking bazel.

The `bazel build` command builds a script named build_pip_package. Running this script as follows will build a .whl file within the /tmp/tensorflow_pkg directory:
```
$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

#### Install the pip package

Invoke pip install to install that pip package. The filename of the .whl file depends on your platform. For example, the following command will install the pip package

```
$ sudo pip install /tmp/tensorflow_pkg/tensorflow-[detailed version].whl
```

## Uninstalling TensorFlow

To uninstall TensorFlow, issue one of following commands:
```
$ sudo pip uninstall tensorflow  # for Python 2.7
$ sudo pip3 uninstall tensorflow # for Python 3.n
```

## Install OpenAI gym
* You can perform a minimal install of gym with:
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```
If you prefer, you can do a minimal install of the packaged version directly from PyPI:
```
pip install gym
```
You'll be able to run a few environments right away:
```
algorithmic
toy_text
classic_control (you'll need pyglet to render though)
```
We recommend playing with those environments at first, and then later installing the dependencies for the remaining environments.

* Installing everything

To install the full set of environments, you'll need to have some system packages installed. We'll build out the list here over time; please let us know what you end up installing on your platform.

On OSX:
```
brew install cmake boost boost-python sdl2 swig wget
```

On Ubuntu 14.04:
```
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
MuJoCo has a proprietary dependency we can't set up for you. Follow the instructions in the mujoco-py package for help.

Once you're ready to install everything, run pip install -e '.[all]' (or pip install 'gym[all]').
