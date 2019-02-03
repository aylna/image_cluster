#Official Tensorflow Docker image https://hub.docker.com/r/tensorflow/tensorflow/
FROM tensorflow/tensorflow:latest-py3

WORKDIR /app
ADD . /app

RUN python3 -m pip install --upgrade setuptools
RUN python3 -m pip install -r requirements.txt

CMD [ "python3", "./autoencoder_fonks.py" ]


