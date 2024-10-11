# conversion/tvm_conversion.py
import os
import subprocess

#Due to TVM not installable on conda environment executed as docker container
def run_tvm_docker_conversion(model_path, framework, input_shape, target, output_dir):
    docker_command = [
        "docker", "run", "--rm", "-v", f"{os.getcwd()}:/workspace", "-w", "/workspace", "tlcpack/ci-cpu",
        "python", "tvm_conversion.py",
        "--model_path", model_path,
        "--framework", framework,
        "--input_shape", input_shape,
        "--target", target,
        "--output_dir", output_dir
    ]

    try:
        result = subprocess.run(docker_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Error during TVM conversion: {e.stderr.decode('utf-8')}")