FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget unzip libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
    
RUN git clone https://github.com/kohya-ss/sd-scripts /app/kohya_ss

RUN pip install --upgrade pip && \
    cd /app/kohya_ss && pip install -r requirements.txt

COPY . /app/
RUN pip uninstall opencv-python -y


RUN pip install -r /app/requirements.txt


CMD python3 -u handler.py

