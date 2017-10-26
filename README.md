# Installation note for set-up Tensorflow
## Install Bazel
* Install JDK 8
Install JDK 8 by using:

```sudo apt-get install openjdk-8-jdk```

On Ubuntu 14.04 LTS you'll have to use a PPA:

```
sudo add-apt-repository ppa:webupd8team/java'
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
