#!/bin/sh

# uidは自分の計算機の/etc/passwdを見るなどして変更する
uid=1002
image=evaluate2

docker run --rm -it -v /tmp:/tmp -v /mnt/:/mnt -v /home:/home \
     -v /media/:/media -v /etc/group:/etc/group:ro \
     -v /etc/passwd:/etc/passwd:ro -u ${uid}:${uid} --gpus all \
     $image bash
