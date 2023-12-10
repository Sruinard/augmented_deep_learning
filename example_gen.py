"""
This file provides the code to generate TFRecords from the raw data.
"""
from typing import Dict
import tensorflow as tf
import os

from absl import app, flags, logging

FLAGS = flags.FLAGS
# add src, dst and train_split_fraction as flags
flags.DEFINE_string("src", None, "Path to the folder containing the raw data")
flags.DEFINE_string("dst", None, "Path to the folder containing the tfrecords")
flags.DEFINE_float("train_fraction", 0.8, "Fraction of the train dataset")


def read_csv(path):
    record_defaults = [tf.float32] * 30 + [tf.int32]
    return tf.data.experimental.CsvDataset(
        path, record_defaults=record_defaults, header=True
    )


def read_csv_from_folder(src):
    creditcard_files = tf.data.Dataset.list_files(os.path.join(src, "*.csv"))
    creditcard_ds = creditcard_files.interleave(read_csv, cycle_length=3)
    return creditcard_ds


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def split_dataset_in_train_test(ds, train_split_fraction=0.8):
    # this is a bit ugly, but textline datasets returns unknown cardinatility
    logging.info("Counting elements in dataset")
    n_elems_in_ds = len(list(ds.as_numpy_iterator()))
    ds = ds.take(n_elems_in_ds)  # set cardinatility
    logging.info(f"Dataset contains {n_elems_in_ds} elements")

    # build train and test dataset
    train_size = int(n_elems_in_ds * train_split_fraction)
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size)
    return {"train": train_ds, "test": test_ds}


# load, split and write to tfrecords
def write_datasets_to_tfrecord(datasets: Dict[str, tf.data.Dataset], dst: str):
    """
    Write datasets to tfrecords
    """

    if not os.path.exists(dst):
        logging.info(f"Creating folder {dst}")
        os.makedirs(dst)

    for name, ds in datasets.items():
        logging.info(f"Writing dataset {name} to tfrecord")
        writer = tf.io.TFRecordWriter(os.path.join(dst, f"{name}.tfrecord"))
        for row in ds:
            feature = {
                "V1": _floats_feature(row[1]),
                "V2": _floats_feature(row[2]),
                "V3": _floats_feature(row[3]),
                "V4": _floats_feature(row[4]),
                "V5": _floats_feature(row[5]),
                "V6": _floats_feature(row[6]),
                "V7": _floats_feature(row[7]),
                "V8": _floats_feature(row[8]),
                "V9": _floats_feature(row[9]),
                "V10": _floats_feature(row[10]),
                "V11": _floats_feature(row[11]),
                "V12": _floats_feature(row[12]),
                "V13": _floats_feature(row[13]),
                "V14": _floats_feature(row[14]),
                "V15": _floats_feature(row[15]),
                "V16": _floats_feature(row[16]),
                "V17": _floats_feature(row[17]),
                "V18": _floats_feature(row[18]),
                "V19": _floats_feature(row[19]),
                "V20": _floats_feature(row[20]),
                "V21": _floats_feature(row[21]),
                "V22": _floats_feature(row[22]),
                "V23": _floats_feature(row[23]),
                "V24": _floats_feature(row[24]),
                "V25": _floats_feature(row[25]),
                "V26": _floats_feature(row[26]),
                "V27": _floats_feature(row[27]),
                "V28": _floats_feature(row[28]),
                "Amount": _floats_feature(row[29]),
                "Class": _int64_feature(row[30]),
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()


def main(argv):
    del argv  # unused
    # load data
    creditcard_ds = read_csv_from_folder(FLAGS.src)

    # split dataset
    datasets = split_dataset_in_train_test(creditcard_ds, FLAGS.train_fraction)

    # write datasets to tfrecords
    write_datasets_to_tfrecord(datasets, FLAGS.dst)


if __name__ == "__main__":
    app.run(main)
