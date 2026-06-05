FROM continuumio/miniconda3

WORKDIR /app

RUN conda create -n miccai python=3.7 -y

SHELL ["conda", "run", "-n", "miccai", "/bin/bash", "-c"]

RUN pip install pip==21.3.1

RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt /app/
RUN pip install -r requirements.txt