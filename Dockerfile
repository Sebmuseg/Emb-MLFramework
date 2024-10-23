# Stage 1: Build environment
#Final Image around 5GB 
FROM python:3.13-slim AS builder

# Install system dependencies and wget to download Miniconda
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda for ARM architecture (adapt for x86 if necessary)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Make sure conda is available in the path
ENV PATH="/opt/conda/bin:$PATH"

# Configure conda to use the classic solver instead of libmamba
RUN conda config --set solver classic

# Install Core Machine Learning Libraries with Conda
RUN conda install -c conda-forge lightgbm
RUN conda install -c conda-forge onnx
RUN conda install -c conda-forge onnxruntime
RUN conda install -c conda-forge scikit-learn
RUN conda install -c conda-forge tensorflow
RUN conda install -c conda-forge pytorch
RUN conda install -c conda-forge xgboost

# Check for architecture and install openvino only for x86_64
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        conda install -c conda-forge openvino openvino-dev; \
    fi

# Install Utility Libraries with Conda
RUN conda install -c conda-forge paramiko
RUN conda install -c conda-forge protobuf
RUN conda install -c conda-forge urllib3
RUN conda install -c conda-forge requests

# Install the remaining dependencies with pip (from your requirements file)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Create the final lightweight image
FROM python:3.13-slim

# Copy only the Conda environment and installed dependencies from the builder stage
COPY --from=builder /opt/conda /opt/conda

# Make sure conda is available in the final image
ENV PATH="/opt/conda/bin:$PATH"

# Copy the necessary files for the application from your local machine
WORKDIR /app
COPY . .

# Optionally, install any remaining Python dependencies with pip if needed (e.g., conversion tools like tf2onnx)
# RUN pip install --no-cache-dir -r requirements.txt

# Set the CMD to launch bash or another command for running your app
CMD ["/bin/bash"]