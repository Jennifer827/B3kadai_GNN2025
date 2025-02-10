FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ "Asia/Tokyo"

RUN apt-get --no-install-recommends update && \
    apt-get --no-install-recommends install -y ffmpeg libsm6 libxext6

RUN adduser dev
USER dev

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install ipdb==0.13.13 \
    opencv-python==4.5.5.64 \
    opencv-python-headless==4.5.5.64
