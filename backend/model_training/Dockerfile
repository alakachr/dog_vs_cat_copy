FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

#EXPOSE 8080

CMD ["python", "model_training.py"]