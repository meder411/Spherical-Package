# Spherical Distortion Package

This repository contains the backend code for Mapped Convolutions and Tangent Images. Is it installable as Python package, and is built around PyTorch. The mapped convolution operations as well as the resampling operations for tangent images are written in a mix of C++ and CUDA and wrapped as PyTorch modules,


## Requirements and Installation

To use this package there are a few dependencies. Most of them are Python-based, but there are a few binary package needed as well.


### Quick-start with Docker 

A [`Dockerfile`](https://github.com/meder411/spherical-package/blob/master/Dockerfile) is provided for any use who have Docker set up. This file assumes that you have `nvidia-docker` set up.

To build the docker container, run:

```
bash docker_build.sh
```

To run the environment, modify `docker_run.sh` as needed, and then run:

```
bash docker_run.sh
```

### Manual set up

This package requires 3 non-Python dependencies: [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [CGAL](https://www.cgal.org/), and [OpenEXR](https://www.openexr.com/).

Unfortunately, the most recent version of Eigen as of this release (3.3.7) doesn't support some GPU operations. As a result, you'll need to install Eigen from source to run the distortion module.

To install Eigen from source, first navigate to the desired directory and then run the following commands:

```
git clone https://gitlab.com/libeigen/eigen.git \
    && cd eigen \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install
```

CGAL and OpenEXR can be installed via aptitude:

```
sudo apt install libcgal-dev libopenexr-dev
```

Once these packages are installed, you can install the Python dependencies with:

```
pip install -r requirements.txt
```

Finally, install this package with the commands:

```
python setup.py build -j 4
python setup.py install
```
Note that you can adjust the number after  `-j`  according to the number of cores available to your machine. It parallelizes compilation.