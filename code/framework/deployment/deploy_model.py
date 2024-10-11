# deployment/deploy_model.py
import paramiko  # For SSH connection to the device
import os
import subprocess


def deploy_model_to_device(model_name, model_path, device_ip, device_username, deployment_path, model_format):
    """
    Deploy a model from the framework to a target embedded/edge device via SSH.

    Parameters:
    - model_name: The name of the model to deploy.
    - model_path: The local path to the model file.
    - device_ip: The IP address of the target device.
    - device_username: The username for SSH login to the target device.
    - deployment_path: The file path on the target device where the model will be deployed.
    - model_format: The format of the model (e.g., ONNX, OpenVINO, TensorFlow Lite).

    Returns:
    - success: Boolean indicating whether the deployment was successful.
    """
    try:
        # Set up SSH client to connect to the target device
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(device_ip, username=device_username)

        # Set up SFTP to transfer the model file
        sftp = ssh.open_sftp()

        # Transfer model to the device
        print(f"Transferring {model_name} to {device_ip}:{deployment_path}")
        sftp.put(model_path, os.path.join(deployment_path, os.path.basename(model_path)))

        # Verify transfer and deploy (optional: execute deployment script on device)
        print(f"Model {model_name} transferred successfully.")
        
        # Run any device-specific setup or inference engine setup here if necessary
        # For example: setting up OpenVINO runtime, TensorFlow Lite, etc.
        
        # Close SFTP and SSH connection
        sftp.close()
        ssh.close()

        print(f"Model {model_name} successfully deployed to {device_ip}.")
        return True

    except Exception as e:
        print(f"Failed to deploy model to device: {e}")
        return False
    

def package_model_in_docker(model_name, model_path, dockerfile_template, output_dir):
    """
    Packages the model into a Docker container.

    Parameters:
    - model_name: The name of the model.
    - model_path: Path to the model file (e.g., ONNX model).
    - dockerfile_template: Path to the Dockerfile template.
    - output_dir: Directory to store the built Docker image.

    Steps:
    - Copy the Dockerfile template and model to the output directory.
    - Build the Docker image.
    - Tag the image with the model name.
    """
    try:
        # Prepare the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy the Dockerfile template and model to the output directory
        dockerfile_path = os.path.join(output_dir, "Dockerfile")
        subprocess.run(["cp", dockerfile_template, dockerfile_path], check=True)
        subprocess.run(["cp", model_path, os.path.join(output_dir, "model.onnx")], check=True)

        # Build the Docker image
        build_command = ["docker", "build", "-t", model_name, output_dir]
        subprocess.run(build_command, check=True)

        print(f"Model {model_name} successfully packaged into a Docker container.")
    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}")