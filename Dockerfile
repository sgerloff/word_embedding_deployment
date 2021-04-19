#FROM python:3.8
FROM tensorflow/tensorflow:2.4.1
WORKDIR /
COPY src /src
COPY data /data
COPY requirements.txt /
RUN pip install -r requirements.txt
CMD ["python", "-m", "src.server"]