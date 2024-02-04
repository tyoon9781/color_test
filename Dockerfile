FROM python:3.10.13-slim

WORKDIR /code

# python library install
RUN pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install fastapi==0.109.1
RUN pip install uvicorn==0.27.0

