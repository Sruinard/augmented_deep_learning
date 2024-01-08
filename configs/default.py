"""Default Hyperparameter configuration."""

import os
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.train_fraction = 0.8
    config.raw_data_src = "./data/raw"

    config.artifact_dir = "./artifacts"
    config.example_gen_dir = os.path.join(config.artifact_dir, "example_gen")
    config.train_src = os.path.join(config.example_gen_dir, "train.tfrecord")
    config.val_src = os.path.join(config.example_gen_dir, "eval.tfrecord")
    config.logdir = os.path.join(config.artifact_dir, "logs")
    config.checkpoint_dir = os.path.abspath(
        os.path.join(config.artifact_dir, "checkpoints")
    )
    config.model_serving_dir = os.path.abspath(
        os.path.join(config.artifact_dir, "models")
    )

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.seed = 42

    config.n_epochs = 10
    config.n_steps_per_epoch = 1000
    config.n_train_steps = 1000
    config.n_eval_steps = 100

    config.model_name = "creditcard"
    return config


def metrics():
    return []
