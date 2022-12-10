FROM nvidia/cuda:11.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG python_version="3.7.3"
ARG OPENCV_VERSION="4.5.3"

# like tool
RUN apt-get update && apt-get -y upgrade &&\
    apt-get install -y --no-install-recommends \
    vim unzip byobu wget tree git cmake\
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# install python3.7.3
RUN apt-get update && apt-get install -y zlib1g-dev libssl-dev libffi-dev build-essential \
    # checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev \
    checkinstall libncursesw5-dev libssl-dev libsqlite3-dev tk-dev \
    libgdbm-dev libc6-dev libbz2-dev &&\
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN mkdir /workspace
WORKDIR /workspace

# install python3
RUN wget -c https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tgz
RUN    tar zxvf Python-${python_version}.tgz
WORKDIR Python-${python_version} 
RUN    ./configure --enable-optimizations --enable-shared CFLAGS=-fPIC
# RUN    make -j8
RUN make install
RUN ldconfig
RUN cd .. 
# RUN rm Python-${python_version}.tgz
RUN    rm -rf Python-${python_version}
RUN pip3 install -U pip
RUN pip3 install -U setuptools

# install opencv
# RUN pip3 install -U numpy
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc g++ libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev \
#     libgtk-3-dev &&\
#     rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# RUN wget -c https://github.com/opencv/opencv/archive/${version}.tar.gz &&\
#     tar -zxvf ${version}.tar.gz && rm ${version}.tar.gz
# RUN mkdir /workspace/opencv-${version}/build

# ## opencv_contrib
# RUN mkdir /workspace/opencv_contrib
# WORKDIR /workspace/opencv_contrib
# RUN wget -c https://github.com/opencv/opencv_contrib/archive/${version}.tar.gz &&\
#     tar -zxvf ${version}.tar.gz && rm /workspace/opencv_contrib/${version}.tar.gz
# WORKDIR /workspace/opencv-${version}/build

# ## make opencv
# RUN cmake -DBUILD_opencv_gapi:BOOL=OFF -DPYTHON_DEFAULT_EXECUTABLE=$(which python3)　-OPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-${version}/modules ..
# RUN make -j8 && make install && ldconfig

RUN apt-get update && apt-get upgrade -y &&\
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        # python3-dev \
        # python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# RUN pip3 install -U pip
# RUN pip3 install -U setuptools

WORKDIR /workspace/
    # Download and unzip OpenCV and opencv_contrib and delte zip files
RUN wget -c https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz &&\
    tar -zxvf ${OPENCV_VERSION}.tar.gz && rm ${OPENCV_VERSION}.tar.gz
RUN wget -c https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz &&\
    tar -zxvf ${OPENCV_VERSION}.tar.gz && rm ${OPENCV_VERSION}.tar.gz

# RUN    wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip
# RUN    unzip ${OPENCV_VERSION}.zip
# RUN    rm ${OPENCV_VERSION}.zip
# RUN    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip
# RUN   unzip ${OPENCV_VERSION}.zip
# RUN    rm ${OPENCV_VERSION}.zip
    # Create build folder and switch to it
# RUN    mkdir /workspace/opencv-${OPENCV_VERSION}
RUN    mkdir /workspace/opencv-${OPENCV_VERSION}/build
WORKDIR /workspace/opencv-${OPENCV_VERSION}/build
    # Cmake configure
RUN    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/workspace/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. 
    # Make
RUN    make -j"$(nproc)"
    # Install to /usr/local/lib
RUN    make install
RUN    ldconfig
    # Remove OpenCV sources and build folder
RUN    rm -rf /workspace/opencv-${OPENCV_VERSION}
RUN rm -rf /workspace/opencv_contrib-${OPENCV_VERSION}


WORKDIR /
RUN rm -rf workspace/


# Docker内で
WORKDIR /home