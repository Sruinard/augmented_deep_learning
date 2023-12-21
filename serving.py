import os
import glob
import tensorflow as tf
import input_pipeline as ip
import json
import requests
import subprocess
import time

raw_batch = {
    "V1": tf.constant([-1.359807]),
    "V2": tf.constant([-0.072781]),
    "V3": tf.constant([2.536347]),
    "V4": tf.constant([1.378155]),
    "V5": tf.constant([-0.338321]),
    "V6": tf.constant([0.462388]),
    "V7": tf.constant([0.239599]),
    "V8": tf.constant([0.098698]),
    "V9": tf.constant([0.363787]),
    "V10": tf.constant([0.090794]),
    "V11": tf.constant([-0.5516]),
    "V12": tf.constant([-0.617801]),
    "V13": tf.constant([-0.99139]),
    "V14": tf.constant([-0.311169]),
    "V15": tf.constant([1.468177]),
    "V16": tf.constant([-0.4704]),
    "V17": tf.constant([0.207971]),
    "V18": tf.constant([0.02579]),
    "V19": tf.constant([0.40399]),
    "V20": tf.constant([0.251412]),
    "V21": tf.constant([-0.018307]),
    "V22": tf.constant([0.277838]),
    "V23": tf.constant([-0.110474]),
    "V24": tf.constant([0.066928]),
    "V25": tf.constant([0.128539]),
    "V26": tf.constant([-0.189115]),
    "V27": tf.constant([0.133558]),
    "V28": tf.constant([-0.021053]),
    "Amount": tf.constant([149.62]),
}


def load_latest_model(serving_model_dir: str, model_name: str):
    models_dir = os.path.join(serving_model_dir, model_name)
    latest_version = max([int(v) for v in os.listdir(models_dir)])
    model_path = os.path.join(models_dir, str(latest_version))
    return tf.saved_model.load(model_path)


def predict(model, batch):
    return model.signatures["serving_default"](**batch)["output_0"]


def stop_and_remove_container(container_name):
    try:
        # Build the Docker command to stop and remove the container
        docker_command = f"docker rm -f {container_name}"

        # Run the Docker command using subprocess
        subprocess.run(docker_command, shell=True, check=True)

        print(f"Container {container_name} stopped and removed.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def run_tf_serving(abs_model_src: str):
    # check if mac_m1
    is_mac_m1 = False
    if "arm64" in subprocess.check_output("uname -a", shell=True).decode("utf-8"):
        is_mac_m1 = True

    docker_image = "emacski/tensorflow-serving" if is_mac_m1 else "tensorflow/serving"

    # stop container if running
    stop_and_remove_container("creditcard")

    time.sleep(5)

    #  docker run  -p 8501:8501 --name creditcard --mount type=bind,source=/Users/stefruinard/Documents/personal/projects/202312_probabilistic_deep_learning/augmented_deep_learning/models/saved_model/creditcard,target=/models/creditcard -e MODEL_NAME=creditcard -t emacski/tensorflow-serving
    subprocess.Popen(
        [
            "docker",
            "run",
            "-p",
            "8501:8501",
            "--name",
            "creditcard",
            "--mount",
            f"type=bind,source={abs_model_src},target=/models/creditcard",
            "-e",
            "MODEL_NAME=creditcard",
            "-t",
            docker_image,
        ]
    )
    time.sleep(5)


def predict_with_docker(model_name, batch, use_instances_key=True):
    # run docker container

    url_sig = f"http://localhost:8501/v1/models/{model_name}:predict"
    headers = {"content-type": "application/json"}
    data = json.dumps(
        {
            "signature_name": "serving_default",
            "instances": [
                {k: v.numpy().tolist()[0] for k, v in batch.items()},
                {k: v.numpy().tolist()[0] for k, v in batch.items()},
            ],
        }
    )
    if not use_instances_key:
        data = json.dumps(
            {
                "signature_name": "serving_default",
                "inputs": {k: v.numpy().tolist() for k, v in batch.items()},
            }
        )

    json_response = requests.post(url_sig, data=data, headers=headers)
    return json.loads(json_response.text)


def main():
    model_name = "creditcard"
    serving_model_dir = "models/saved_model"
    model = load_latest_model(serving_model_dir, model_name)
    pred = predict(model, raw_batch)
    print(pred)

    run_tf_serving(os.path.abspath(os.path.join(serving_model_dir, model_name)))
    pred = predict_with_docker(model_name, raw_batch)
    print(pred)


if __name__ == "__main__":
    main()
