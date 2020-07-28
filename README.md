# Spherical Distortion Package

**Version: 1.0.3**

This repository contains the backend code for [Mapped Convolutions](https://github.com/meder411/MappedConvolutions) and [Tangent Images](https://github.com/meder411/Tangent-Images). Is it installable as Python package, and is built around PyTorch. The mapped convolution operations as well as the resampling operations for tangent images are written in a mix of C++ and CUDA and wrapped as PyTorch modules.


## Requirements and Installation

To use this package there are a few dependencies. Most of them are Python-based, but there are a few binary package needed as well.

### Modifying `setup.py`

Before you go about installing anything, you can tweak a few parameters in [`setup.py`](./setup.py). Namely:

* `module`: should be either `'tangent-images'` or `'mapped-conv'`. The former will only install the necessary backend for the tangent images project, while the latter will install everything. Defaults to `mapped-conv`.
* `use_ninja`: defaults to `False`, but you can set it to `True` if you want to install using the faster `ninja` build system instead of the standard `setuptools`.
* `compute_arch`: should be changed according to your GPU settings. The default should work for most modern GPUs.
* `additional_includes`: use this if you install any of the dependencies to unsual locations.


### Quick-start with Docker 

A [`Dockerfile`](https://github.com/meder411/spherical-package/blob/master/Dockerfile) is provided for any users who have Docker set up. This file assumes that you have `nvidia-docker` set up.

To build the docker container, run:

```
bash docker_build.sh
```

To run the environment, modify `docker_run.sh` as needed, and then run:

```
bash docker_run.sh
```

Note that when using the Docker setup, the contents of this repo are copied to `/package` inside the Docker container.

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

Once these packages are installed, you can should the Python dependencies. I recommend using a conda or virtualenv environment at the very least. All of this code expects Python3.6+ and PyTorch >1.4.0. Once you have your environment activated, install the Python dependencies with:

```
pip install -r requirements.txt
```

Finally, install this package with the commands:

```
python setup.py build -j 4
python setup.py install
```
Note that you can adjust the number after  `-j`  according to the number of cores available to your machine. It parallelizes compilation.


## Testing the Install

Not everything is unit tested, but there is a suite of unit tests associated with the mapped convolution operations and the resampling operations, which underpin most of the other code. To ensure that everything is set up properly, you can run:

```
pytest tests/
```

All 52 tests should pass. If CUDA is not available on your system, the CUDA tests will be skipped. In that case the output should read "26 passed, 26 skipped."

For any further tests, you should run the examples in whichever application repository you are using ([Mapped Convolutions](https://github.com/meder411/MappedConvolutions) or [Tangent Images](https://github.com/meder411/Tangent-Images)).
