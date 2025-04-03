ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Run apt-get (dkpg) without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES all

########################################
# SECTION 1: Essentials                #
########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget unzip pkg-config \
    cmake checkinstall build-essential ca-certificates \
    software-properties-common apt-utils bash-completion \
    rsync tmux \
    # Vision-related libraries \
    ffmpeg htop\
    # OpenGL/glvnd libraries \
    libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 libjpeg-dev libpng-dev \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 libvulkan1\
    libsm6 libglfw3-dev libgles2-mesa-dev \
    xvfb \
    # Default python with pip \
    python3 python3-dev python3-pip \
    # Tkinter GUI \
    python3.10-tk \
    # Setup locale language config \
    locales \
  && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/* \
  # Upgrade to up-to-date pip \
  && python3 -m pip install --upgrade pip \
  # Setup locale language config \
  && locale-gen "en_US.UTF-8" \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  # Set timezone \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && curl -O https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh \
  && bash script.deb.sh \ 
  && apt-get install git-lfs -y
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8


##################################################
# SECTION 2: Install Python Dependencies w/ pip  #
##################################################
ARG PIP_REQ_PKG_DIR="."

COPY "${PIP_REQ_PKG_DIR}/requirements.txt" /tmp
RUN python3 -m pip install -r /tmp/requirements.txt \
    && rm -r /tmp/*
    
# RUN python3 -m pip install https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20241003+c877626f-cp310-cp310-manylinux2014_x86_64.whl \
RUN python3 -m pip install -U 'sapien>=3.0.0b1' 

# download physx GPU binary via sapien
RUN python3 -c "exec('import sapien.physx as physx;\ntry:\n  physx.enable_gpu()\nexcept:\n  pass;')"


# ##################################################
# # SECTION 3: Install dex-retargeting             #
# ##################################################
ARG DEX_RETARGETING_WORKDIR="/dex_retargeting"

# # Download other dependencies
RUN git clone https://github.com/dexsuite/dex-retargeting.git --recursive "${DEX_RETARGETING_WORKDIR}" 
#   && cd "${EMD_WORKDIR}/assets"

##################################################
# SECTION 3: Install emd                         #
##################################################
ARG EMD_WORKDIR="/emd"

# Download other dependencies
RUN git clone https://github.com/Colin97/MSN-Point-Cloud-Completion.git "${EMD_WORKDIR}" 


########################################
# SECTION N: Additional config & MISC  #
########################################
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bin/bash"]
