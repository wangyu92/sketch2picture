FROM nvcr.io/nvidia/pytorch:23.07-py3

# set work directory
WORKDIR /root/src

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV CUDA_DEVICE_ORDER PCI_BUS_ID

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# install system dependencies
RUN rm -f /etc/apt/sources.list.d/archive_uri-*
RUN apt update -y
RUN apt-get install -y --no-install-recommends \
    openssh-server \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    python3-dev \
    python3-numpy \
    libavcodec-dev \
    libavcodec-extra \
    libavformat-dev \
    libswscale-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    libgtk-3-dev \
    libpng-dev \
    libjpeg-dev \
    libopenexr-dev \
    libtiff-dev \
    libwebp-dev \
    libopencv-dev \
    libjasper-dev \
    x264 \
    libx264-dev \
    libssl-dev \
    ffmpeg \
    && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --upgrade pip
COPY ./requirements.txt /root/src/requirements.txt
RUN pip install -r requirements.txt
RUN python -m pip install --no-binary opencv-python opencv-python

# Configure SSH
RUN echo 'root:{}' | chpasswd
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config  # I don't know the exact reason, but you need to make it for external access
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
RUN echo 'export NOTVISIBLE="in users profile"' >> ~/.bashrc
RUN echo "export VISIBLE=now" >> /etc/profile

# copy public key
RUN mkdir -p /root/.ssh
COPY ./*.pub /root/.ssh/
RUN cat /root/.ssh/*.pub >> /root/.ssh/authorized_keys

# Install opencv
RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D BUILD_EXAMPLES=ON .. && \
    make -j4 && \
    make install && \
    ldconfig

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
