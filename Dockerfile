FROM python:3.8
RUN useradd -ms /bin/bash admin
USER admin
WORKDIR /app
COPY app /app
RUN pip install -r requirements.txt
CMD ["ls"]
CMD ["python", "server.py"]