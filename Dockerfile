# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.9

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app
