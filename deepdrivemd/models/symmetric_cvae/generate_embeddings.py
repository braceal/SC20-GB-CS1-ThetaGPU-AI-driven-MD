import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

from deepdrivemd.models.symmetric_cvae.model import model_fn
from deepdrivemd.models.symmetric_cvae.utils import write_single_tfrecord
from deepdrivemd.models.symmetric_cvae.data import parse_function_record_predict

logger = logging.getLogger(__name__)


def generate_embeddings(
    h5_files: List[str],
    checkpoint_dir: str,
    tfrecords_dir: str,
    input_shape: List[int] = [1, 28, 28],
    final_shape: List[int] = [1, 28, 28],
    tfrecord_shape: List[int] = [1, 28, 28],
    predict_batch_size: int = 128,
    mixed_precision: bool = True,
    full_precision_loss: bool = False,
    reconstruction_loss_reduction_type: str = "sum",
    kl_loss_reduction_type: str = "sum",
    model_random_seed: Optional[int] = None,
    loss_scale: float = 1.0,
    metrics: bool = False,
    enc_conv_kernels: List[int] = [5, 5, 5, 5],
    enc_conv_filters: List[int] = [100, 100, 100, 100],
    enc_conv_strides: List[int] = [1, 1, 1, 1],
    dec_conv_kernels: List[int] = [5, 5, 5, 5],
    dec_conv_filters: List[int] = [100, 100, 100, 100],
    dec_conv_strides: List[int] = [1, 1, 1, 1],
    dense_units: int = 64,
    latent_ndim: int = 10,
    activation: str = "relu",
):

    Path(tfrecords_dir).mkdir(exist_ok=True)
    # Convert HDF5 to tfrecords
    for h5_file in h5_files:
        write_single_tfrecord(h5_file, input_shape[1:], final_shape[1:], tfrecords_dir)

    # Get list of tfrecord paths
    tfrecord_files = [
        Path(tfrecords_dir).joinpath(Path(f).with_suffix(".tfrecords").name).as_posix()
        for f in h5_files
    ]

    # Use files closure to get correct data sample
    def _data_generator():
        dtype = tf.float16 if mixed_precision else tf.float32
        list_files = tf.data.Dataset.list_files(tfrecord_files)
        dataset = tf.data.TFRecordDataset(list_files)

        # TODO: We want drop_remainder=False but this needs to be rewritten:
        dataset = dataset.batch(predict_batch_size, drop_remainder=True)
        parse_sample = parse_function_record_predict(dtype, tfrecord_shape, input_shape)
        return dataset.map(parse_sample)

    # Get latest model weights
    weights_file = tf.train.latest_checkpoint(checkpoint_dir)

    print("weights file:", weights_file)

    # Setup model_fn params
    params = {
        "input_shape": input_shape,
        "enc_conv_kernels": enc_conv_kernels,
        "enc_conv_filters": enc_conv_filters,
        "enc_conv_strides": enc_conv_strides,
        "dec_conv_kernels": dec_conv_kernels,
        "dec_conv_filters": dec_conv_filters,
        "dec_conv_strides": dec_conv_strides,
        "dense_units": dense_units,
        "latent_ndim": latent_ndim,
        "activation": activation,
        "mixed_precision": mixed_precision,
        "full_precision_loss": full_precision_loss,
        "reconstruction_loss_reduction_type": reconstruction_loss_reduction_type,
        "kl_loss_reduction_type": kl_loss_reduction_type,
        "model_random_seed": model_random_seed,
        "loss_scale": loss_scale,
        "metrics": metrics,
    }

    tf_config = tf.estimator.RunConfig()
    est = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=tf_config,
    )
    gen = est.predict(
        input_fn=_data_generator,
        checkpoint_path=weights_file,
        yield_single_examples=True,
    )
    return np.array([list(it.values())[0] for it in gen])

if __name__ == "__main__":
    h5_path = "/homes/abrace/data/bba/deepdrivemd_runs/bba_28_cs1.2/bba_28_cs1.2_h5/"
    h5_files = sorted(map(str, Path(h5_path).glob("*.h5")))

    embeddings = generate_embeddings(
        h5_files=h5_files,
        checkpoint_dir="/homes/abrace/data/bba/deepdrivemd_runs/bba_28_cs1.2/model_weights",
        tfrecords_dir="/homes/abrace/tmp/bba_28_cs1.2_analysis/stream-ai-md-tfrecords",
        predict_batch_size=200,
    )

    print(embeddings.shape)
    np.save("/homes/abrace/tmp/bba_28_cs1.2_analysis/stream-ai-md-embeddings.npy", embeddings)

