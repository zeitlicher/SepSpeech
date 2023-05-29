# 必要な場合はproxy設定を追加する
# nvidia-driver >= 515 && pytorch-2.0 binary-build
ARG CUDA="11.8.0"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
# rootで実行
USER root
RUN apt-get update
RUN apt-get install -y --no-install-recommends emacs python3.9 python3-pip libsndfile1-dev
# unfortunatelly, we do not use CUDA 12.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torchmetrics lightning einops numpy sklearn tqdm pandas scipy numba h5py seaborn wave librosa
RUN pip3 install tensorboard
RUN pip3 install git+https://github.com/openai/whisper.git
RUN pip3 install torchmetrics[audio], ffmpeg
ARG GIT_VERSION=2.38.0
RUN <<-EOF
	set -eu
  # 必要なパッケージをインストール
	apt-get update -qq
	apt-get install -qy libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev
	cd /usr/local/src
  # Gitのソースコードをリポジトリからダウンロード
	wget https://github.com/git/git/archive/refs/tags/v${GIT_VERSION}.tar.gz
  # ビルド
	tar -xzf v${GIT_VERSION}.tar.gz
	cd git-${GIT_VERSION}
	make prefix=/usr/local all -j "$(nproc)"
	make prefix=/usr/local install
EOF
