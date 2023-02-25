FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
ENV HTTP_PROXY "http://gw.a.tsukuba-tech.ac.jp:8080"
ENV HTTPS_PROXY "http://gw.a.tsukuba-tech.ac.jp:8080"
# nvidia-driver >= 515
ARG CUDA="11.7.1"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
RUN apt-get -y install emacs python3.9 python3-pip libsndfile1-dev
RUN pip3 install torch torchvision torchaudio torchmetrics lightning
RUN pip3 install einops numpy sklearn tqdm pandas scipy numba h5py seaborn wave
RUN pip3 install librosa