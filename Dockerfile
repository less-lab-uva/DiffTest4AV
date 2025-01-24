# Base Image
FROM continuumio/miniconda3

# Install dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 curl -y

# Copy project folder
RUN mkdir /difftest
COPY ./conda_details_docker.txt /difftest/conda_details_docker.txt
WORKDIR /difftest

# Create difftest environment
RUN conda create --name difftest --file conda_details_docker.txt python-3.8.19

# Activate the 'difftest' environment and install requirements
RUN echo "conda activate difftest" >> ~/.bashrc

# Set the default command to activate the 'difftest' environment
CMD ["conda", "run", "-n", "difftest", "/bin/bash"]