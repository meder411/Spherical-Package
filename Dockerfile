FROM nvidia/cuda:latest

# Install baseline packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    cmake \
    cmake-curses-gui \
    imagemagick \
    vim \
    screen \
    tmux \
    python3-pip \
    libcgal-dev \
    libopenexr-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Eigen from source (aptitude package version has issues with CUDA)
RUN cd /usr/local/src \
    && git clone https://gitlab.com/libeigen/eigen.git \
    && cd eigen \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j 4

# Install necessary Python libraries
RUN pip3 install torch \
    torchvision \
    numpy \
    matplotlib \
    scikit-image \
    visdom \
    plyfile \
    opencv-python==3.4.2.17 \
    opencv-contrib-python==3.4.2.17 \
    yacs \
    ipdb \
    openexr \
    tqdm \
    pytest

# Install the package
# Note: Not editable--Docker will have to be rebuilt for any changes to the package
COPY . /package
RUN cd /package \
    && python3 setup.py build -j 12 \
    && python3 setup.py install \
    && cd .. \
    && rm -r /package