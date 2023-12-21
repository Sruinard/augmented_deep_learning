import clu
import tensorflow as tf
import jax.numpy as jnp
from dataclasses import dataclass

LABEL_KEY = "Class"

TRAIN_SCHEMA = {
    "V1": tf.io.FixedLenFeature([], tf.float32),
    "V2": tf.io.FixedLenFeature([], tf.float32),
    "V3": tf.io.FixedLenFeature([], tf.float32),
    "V4": tf.io.FixedLenFeature([], tf.float32),
    "V5": tf.io.FixedLenFeature([], tf.float32),
    "V6": tf.io.FixedLenFeature([], tf.float32),
    "V7": tf.io.FixedLenFeature([], tf.float32),
    "V8": tf.io.FixedLenFeature([], tf.float32),
    "V9": tf.io.FixedLenFeature([], tf.float32),
    "V10": tf.io.FixedLenFeature([], tf.float32),
    "V11": tf.io.FixedLenFeature([], tf.float32),
    "V12": tf.io.FixedLenFeature([], tf.float32),
    "V13": tf.io.FixedLenFeature([], tf.float32),
    "V14": tf.io.FixedLenFeature([], tf.float32),
    "V15": tf.io.FixedLenFeature([], tf.float32),
    "V16": tf.io.FixedLenFeature([], tf.float32),
    "V17": tf.io.FixedLenFeature([], tf.float32),
    "V18": tf.io.FixedLenFeature([], tf.float32),
    "V19": tf.io.FixedLenFeature([], tf.float32),
    "V20": tf.io.FixedLenFeature([], tf.float32),
    "V21": tf.io.FixedLenFeature([], tf.float32),
    "V22": tf.io.FixedLenFeature([], tf.float32),
    "V23": tf.io.FixedLenFeature([], tf.float32),
    "V24": tf.io.FixedLenFeature([], tf.float32),
    "V25": tf.io.FixedLenFeature([], tf.float32),
    "V26": tf.io.FixedLenFeature([], tf.float32),
    "V27": tf.io.FixedLenFeature([], tf.float32),
    "V28": tf.io.FixedLenFeature([], tf.float32),
    "Amount": tf.io.FixedLenFeature([], tf.float32),
    "Class": tf.io.FixedLenFeature([], tf.int64),
}

SERVING_SCHEMA = {k: v for k, v in TRAIN_SCHEMA.items() if k != LABEL_KEY}

TRAIN_SIGNATURE_DEF = {
    k: tf.TensorSpec(shape=(None,), dtype=v.dtype) for k, v in TRAIN_SCHEMA.items()
}

SERVING_SIGNATURE_DEF = {
    k: tf.TensorSpec(
        shape=(None,),
        dtype=v.dtype,
        name=k,
    )
    for k, v in SERVING_SCHEMA.items()
}


class Preprocessor(tf.Module):
    """Preprocessing module for a TensorFlow model."""

    def __init__(self, epsilon=0.001):
        """
        Constructor for the Preprocessor.

        Parameters:
          - epsilon (float): A small value added to log-transformed features.
        """
        self.epsilon = epsilon
        self.norm = tf.keras.layers.Normalization()

    def _common_preprocess(self, example):
        x = example.copy()
        x["Amount"] = tf.math.log(x["Amount"] + self.epsilon)
        packed_inputs = tf.stack(
            [x[k] for k in sorted(x.keys()) if k != LABEL_KEY], axis=-1
        )
        return packed_inputs

    def _normalize_features(self, packed_inputs):
        norm_features = self.norm(packed_inputs)
        return tf.clip_by_value(norm_features, -5, 5)

    def fit(self, dataset):
        """
        Fit the normalization layer with the given dataset.

        Parameters:
          - dataset (tf.data.Dataset): The training dataset.
        """

        def _prepare_for_fit(example):
            packed_inputs = self._common_preprocess(example)
            return tf.reshape(packed_inputs, [-1, 1])

        # Apply the common preprocessing and reshape for the normalization layer
        prepared_dataset = dataset.map(_prepare_for_fit)

        # Adapt the normalization layer to the dataset
        self.norm.adapt(prepared_dataset)

    @tf.function(input_signature=[TRAIN_SIGNATURE_DEF])
    def preprocessing_fn(self, train_example):
        packed_inputs = self._common_preprocess(train_example)
        norm_features = self._normalize_features(packed_inputs)
        return norm_features, train_example[LABEL_KEY]

    @tf.function(input_signature=[SERVING_SIGNATURE_DEF])
    def serving_fn(self, serving_example):
        packed_inputs = self._common_preprocess(serving_example)
        norm_features = self._normalize_features(packed_inputs)
        return norm_features


@dataclass
class HParams:
    batch_size: int = 64
    shuffle_buffer_size: int = 10000
    shuffle_seed: int = 42
    prefetch_buffer_size: int = 1000
    n_epochs: int = 10


def get_datasets(preprocessor, train_src, val_src, hp: HParams = HParams()):
    """
    Get the training and validation datasets.

    Parameters:
      - src (str): The path to the training data.

    Returns:
      - train_dataset (tf.data.Dataset): The training dataset.
      - val_dataset (tf.data.Dataset): The validation dataset.
    """
    train_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=train_src,
        batch_size=hp.batch_size,
        features=TRAIN_SCHEMA,
        label_key=LABEL_KEY,
        reader=tf.data.TFRecordDataset,
        shuffle_buffer_size=hp.shuffle_buffer_size,
        shuffle_seed=hp.shuffle_seed,
        sloppy_ordering=True,
        num_epochs=1,
        prefetch_buffer_size=hp.prefetch_buffer_size,
        reader_num_threads=8,
        parser_num_threads=8,
        drop_final_batch=True,
    )

    val_dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=val_src,
        batch_size=hp.batch_size,
        features=TRAIN_SCHEMA,
        label_key=LABEL_KEY,
        reader=tf.data.TFRecordDataset,
        shuffle_buffer_size=hp.shuffle_buffer_size,
        shuffle_seed=hp.shuffle_seed,
        sloppy_ordering=True,
        num_epochs=1,
        prefetch_buffer_size=hp.prefetch_buffer_size,
        reader_num_threads=8,
        parser_num_threads=8,
        drop_final_batch=True,
    )

    preprocessor.fit(train_dataset.map(lambda x, _: x))

    # set cardinality for the datasets, otherwise the iterator will loop indefinitely
    n_elems_in_train = sum(1 for _ in train_dataset)
    n_elems_in_val = sum(1 for _ in val_dataset)

    train_dataset = train_dataset.map(
        lambda x, y: (preprocessor.serving_fn(x), tf.expand_dims(y, axis=-1))
    )
    val_dataset = val_dataset.map(
        lambda x, y: (preprocessor.serving_fn(x), tf.reshape(y, (-1, 1)))
    )

    return (
        train_dataset.take(n_elems_in_train).repeat().as_numpy_iterator(),
        val_dataset.take(n_elems_in_val).repeat().as_numpy_iterator(),
    )
