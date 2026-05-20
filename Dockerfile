FROM projectmonai/monai:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Mount your code at runtime, e.g.:
#   docker run --rm -it -v $(pwd):/app daoct-baseline
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

CMD ["bash"]