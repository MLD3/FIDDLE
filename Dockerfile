FROM python:3.8
WORKDIR /workdir

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY FIDDLE/ ./FIDDLE/
