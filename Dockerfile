FROM projectmonai/monai:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install opencv-python

CMD ["bash"]