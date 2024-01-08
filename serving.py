import json
import os
import subprocess
import time

import ml_collections
import requests
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

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


def stop_and_remove_container(container_name, path_to_docker="/usr/bin/docker"):
    try:
        # Build the Docker command to stop and remove the container
        docker_command = f"{path_to_docker} rm -f {container_name}"

        # Run the Docker command using subprocess
        subprocess.run(docker_command, shell=True, check=True)

        print(f"Container {container_name} stopped and removed.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


def run_tf_serving(abs_model_src: str, path_to_docker="/usr/bin/docker"):
    # check if mac_m1
    is_mac_m1 = False
    if "arm64" in subprocess.check_output("uname -a", shell=True).decode("utf-8"):
        is_mac_m1 = True

    docker_image = "emacski/tensorflow-serving" if is_mac_m1 else "tensorflow/serving"

    # stop container if running
    stop_and_remove_container("creditcard")

    time.sleep(5)

    tf_serving_command = f"/usr/local/bin/docker run -p 8501:8501 --name creditcard --mount type=bind,source={abs_model_src},target=/models/creditcard -e MODEL_NAME=creditcard -t {docker_image}"
    subprocess.Popen(tf_serving_command, shell=True)
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

    json_response = requests.post(url_sig, data=data, headers=headers, timeout=5)
    return json.loads(json_response.text)


def inference(model_serving_dir, model_name, batch):
    prediction_template = """
    ------------------------------
    Creditcard Fraud Detection Predictions:
    Prediction: {}
    With Docker: {}
    ------------------------------
    """

    # load model and predict
    model = load_latest_model(model_serving_dir, model_name)
    pred = predict(model, batch)
    logging.info(prediction_template.format(pred, False))

    # use tf serving to predict
    run_tf_serving(os.path.join(model_serving_dir, model_name))
    pred = predict_with_docker(model_name, batch)
    logging.info(prediction_template.format(pred, True))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )

    cfg = FLAGS.config
    inference(cfg.model_serving_dir, cfg.model_name, raw_batch)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
