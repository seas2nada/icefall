FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update -y

RUN apt-get install -y --no-install-recommends \
        build-essential \
        dialog \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar

RUN apt-get install -y --no-install-recommends \
        sudo \
        zip \
        nano \
        tmux \
        tree \
        mc \
        nodejs \
        npm \
        openssh-server \
        rsync \
        sshfs \
        libsndfile1

RUN apt-get install -y --no-install-recommends \
        zlib1g-dev \
        automake \
        autoconf \
        sox \
        gfortran \
        libtool \
        subversion \
        python3 \
        cmake \
        make \
        flac \
        bc \
        libgl1-mesa-glx

RUN apt-get upgrade -y

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ARG workspace=/home/Workspace
ENV WORKSPACE ${workspace}
RUN mkdir -p ${workspace}
WORKDIR ${workspace}

ARG root_passwd=root
RUN echo "root:${root_passwd}" | chpasswd

ARG user=icefall
ARG group=icefall
ARG uid=911
ARG gid=911
ARG passwd=if2022

RUN groupadd -g ${gid} ${group} \
  && useradd -u ${uid} -g ${gid} -m -s /bin/bash ${user} \
  && echo "${user}:${passwd}" | chpasswd \
  && adduser ${user} sudo

# ==================================================================
# tensorboard
# ------------------------------------------------------------------
EXPOSE 6006

#===============================================================================
# SSH
#
# ref: https://docs.docker.com/engine/examples/running_ssh_service/
#-------------------------------------------------------------------------------
RUN mkdir /var/run/sshd
# RUN echo 'root:THEPASSWORDYOUCREATED' | chpasswd
# RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/.*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config

# ref: https://gist.github.com/marcy-terui/9460706
RUN sed -i "s/.*PasswordAuthentication.*/PasswordAuthentication yes/g" /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# ENV NOTVISIBLE "in users profile"
# RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22

# CMD ["/usr/sbin/sshd", "-D"]

#===============================================================================
# entrypoint
#-------------------------------------------------------------------------------

COPY start.sh /
RUN chmod u+x /start.sh

ENV PUID=5555
ENV PGID=5555
CMD ["/start.sh"]
#-------------------------------------------------------------------------------
