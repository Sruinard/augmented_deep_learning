"""Default Hyperparameter configuration."""

import os
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.learning_rate = 0.1
    config.momentum = 0.9
    config.batch_size = 128
    config.num_epochs = 10
    config.seed = 42

    config.n_epochs = 10
    config.n_steps_per_epoch = 1000
    config.n_train_steps = 1000
    config.n_eval_staps = 100

    config.artifact_dir = "./artifacts"
    config.train_src = os.path.join(config.artifact_dir, "example_gen/train.tfrecord")
    config.val_src = os.path.join(config.artifact_dir, "example_gen/test.tfrecord")
    config.logdir = os.path.join(config.artifact_dir, "logs")
    config.checkpoint_dir = os.path.abspath(
        os.path.join(config.artifact_dir, "checkpoints")
    )
    config.model_serving_dir = os.path.join(config.artifact_dir, "models")
    config.model_name = "creditcard"
    return config


def metrics():
    return []
