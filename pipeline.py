from kfp import dsl
from kfp import client
from kfp.dsl import Dataset, Input, Model, Output
import os

import requests
from kfp import local

# local.init(runner=local.DockerRunner())
local.init(runner=local.SubprocessRunner())


@dsl.component(
    packages_to_install=[
        "requests",
    ]
)
def build_base_dataset_op(raw_ds: Output[Dataset]):
    import requests

    result = requests.get(
        "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    )

    raw_ds.metadata["description"] = "Raw dataset for creditcard fraud detection"
    raw_ds.metadata["version"] = "1.0.0"
    raw_ds.metadata["labels"] = {"type": "tabular", "domain": "finance"}
    raw_ds.metadata["extension"] = ".csv"

    with open(raw_ds.path, "wb") as f:
        f.write(result.content)


@dsl.component(base_image="acrkfdwe99.azurecr.io/mlops/creditcard-ml")
def example_gen_op(
    raw_ds: Input[Dataset],
    example_gen_dir: Output[Dataset],
) -> str:
    from configs import default as cfgs
    import example_gen

    cfg = cfgs.get_config()
    creditcard_ds = example_gen.read_csv_from_folder(raw_ds.path)

    # split dataset
    datasets = example_gen.split_dataset_in_train_eval(
        creditcard_ds, cfg.train_fraction
    )

    # write datasets to tfrecords
    example_gen.write_datasets_to_tfrecord(datasets, example_gen_dir.path)


@dsl.pipeline
def ml_pipeline():
    base_ds_task = build_base_dataset_op()
    example_gen_task = example_gen_op(raw_ds=base_ds_task.outputs["raw_ds"])


# base_ds_task = build_base_dataset_op()
# example_gen_task = example_gen_op(raw_ds=base)
# print(base_ds_task.output)
# assert task.output == "Hello, Stef Ruinard + ./artifacts!"

from kfp import compiler

compiler.Compiler().compile(ml_pipeline, "pipeline.yaml")


USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST = "http://127.0.0.1:8080"  # your istio-ingressgateway pod ip:8080
ENDPOINT = "http://localhost:8080"

session = requests.Session()
response = session.get(ENDPOINT)

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}

data = {"login": "user@example.com", "password": "12341234"}
session.post(response.url, headers=headers, data=data)
session_cookie = session.cookies.get_dict()["authservice_session"]

kfp_client = client.Client(
    host=f"{ENDPOINT}/pipeline",
    namespace=f"{NAMESPACE}",
    cookies=f"authservice_session={session_cookie}",
)

# kfp_client = client.Client(host=endpoint, namespace=)
run = kfp_client.create_run_from_pipeline_package(
    "pipeline.yaml",
    # arguments={
    #     "recipient": "Stef Ruinard",
    # },
)
url = f"{ENDPOINT}/#/runs/details/{run.run_id}"
print(url)
