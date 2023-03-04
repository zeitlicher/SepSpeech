# nvidia-driver >= 515
ARG CUDA="11.7.1"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
ENV HTTP_PROXY "http://gw.a.tsukuba-tech.ac.jp:8080"
ENV HTTPS_PROXY "http://gw.a.tsukuba-tech.ac.jp:8080"
RUN apt-get --no-install-recommends -y install emacs python3.9 python3-pip libsndfile1-dev
RUN pip3 install torch torchvision torchaudio torchmetrics lightning einops numpy sklearn tqdm pandas scipy numba h5py seaborn wave librosa