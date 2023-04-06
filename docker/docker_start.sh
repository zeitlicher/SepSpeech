#!/bin/sh

uid=1002
image=pytorch:base

docker run --rm -it -v /tmp:/tmp -v /mnt/:/mnt -v /home:/home \
     -v /media/:/media -v /etc/group:/etc/group:ro \
     -v /etc/passwd:/etc/passwd:ro -u ${uid}:${uid} --gpus all \
     $image bash
