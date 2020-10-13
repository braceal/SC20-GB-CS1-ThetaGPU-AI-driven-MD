from pathlib import Path
import glob
import tensorflow as tf
from deepdrivemd.models.symmetric_cvae.model import model_fn
from deepdrivemd.models.symmetric_cvae.data import (
    parse_function_record_predict,
    parse_function_record,
)
from deepdrivemd.models.symmetric_cvae.utils import write_to_tfrecords, get_params
import os
import json
import socket

# from deepdrivemd.models.symmetric_cvae.data import simulation_tf_record_input_fn

# Dependent on which node
# PROVIDED_IPS = {""}


def init_gpu_training():

    with open(os.environ["COBALT_NODEFILE"], "r") as f:
        nodefile = f.readlines()

    ips, hostnames = [], []

    for line in nodefile:
        # 172.23.2.189 thetagpu01.i2b.alcf.anl.gov
        split_line = line.split()
        ip, hostname = split_line[0], split_line[1]
        ips.append(ip)
        hostnames.append(hostname)

    current_hostname = socket.gethostname()

    host_id = hostnames.index(current_hostname)

    # task = "chief" if host_id == 0 else "worker"
    task = "worker"

    os.environ["TF_CONFIG"] = json.dumps(
        {"cluster": {"worker": ips}, "task": {"type": task, "index": host_id}}
    )


def update_dataset(params):
    sim_data_dir = params["sim_data_dir"]
    train_data_dir = params["data_dir"]
    eval_data_dir = params["eval_data_dir"]
    tfrecord_shape = params["tfrecord_shape"]
    h5_shape = params["h5_shape"]
    fraction = params["fraction"]
    num_samples = params["samples_per_file"]

    h5_files = glob.glob(f"{sim_data_dir}/*.h5")
    write_to_tfrecords(
        files=h5_files,
        initial_shape=h5_shape[1:],
        final_shape=tfrecord_shape[1:],
        num_samples=num_samples,
        train_dir_path=train_data_dir,
        eval_dir_path=eval_data_dir,
        fraction=fraction,
    )


def data_generator(params):
    dtype = tf.float16
    tfrecord_files = Path(params["data_dir"]).glob("*.tfrecords")
    list_files = tf.data.Dataset.list_files(tfrecord_files)
    dataset = tf.data.TFRecordDataset(list_files)

    # TODO: We want drop_remainder=False but this needs to be rewritten:
    dataset = dataset.batch(params["batch_size"], drop_remainder=True)
    parse_sample = parse_function_record_predict(
        dtype, params["tfrecord_shape"], params["input_shape"],
    )
    return dataset.map(parse_sample)


def simulation_tf_record_input_fn(params):
    batch_size = params["batch_size"]
    input_shape = params["tfrecord_shape"]
    final_shape = params["input_shape"]
    mp = True  # params["mixed_precision"]
    dtype = tf.float16 if mp else tf.float32
    data_dir = params["data_dir"]
    # last_n_files = params["last_n_files"]
    files = list(glob.glob(f"{data_dir}/*.tfrecords"))
    # last_n_files = sorted(files, key=sort_string_numbers)[:last_n_files]
    list_files = tf.data.Dataset.list_files(files)
    dataset = tf.data.TFRecordDataset(list_files)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    parse_sample = parse_function_record(dtype, input_shape, final_shape)
    dataset = dataset.map(parse_sample)
    dataset = dataset.shuffle(10,)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    # Distirbute data set over workers
    # dataset = params["strategy"].experimental_distribute_dataset(dataset)
    # dataset = dataset.shard(1, 0)
    return dataset


def main():

    # Congifure GPU

    # multi node
    # init_gpu_training()
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # Single node / multi-gpu
    # strategy = tf.distribute.MirroredStrategy()

    # Single node / single-gpu
    strategy = None

    # h5_dir = "/lus/theta-fs0/projects/RL-fold/braceal/bba/h5"
    yml_file = "/lus/theta-fs0/projects/RL-fold/braceal/bba/params.yaml"
    weights_dir = (
        "/projects/RL-fold/msalim/production-runs/bba_28_case1-long/cvae_weights"
    )
    tfrecords_dir = "/lus/theta-fs0/projects/RL-fold/braceal/bba/tfrecords"
    # embeddings_file = "/lus/theta-fs0/projects/RL-fold/braceal/bba/bba-anton-embeddings.npy"

    params = get_params(yml_file)
    params["fraction"] = 0
    # params["sim_data_dir"] = h5_dir
    params["data_dir"] = tfrecords_dir
    params["eval_data_dir"] = tfrecords_dir
    params["strategy"] = strategy

    # update_dataset(params)

    print("Update dataset done")

    tf_config = tf.estimator.RunConfig(train_distribute=strategy)

    est = tf.estimator.Estimator(
        model_fn,
        model_dir="/lus/theta-fs0/projects/RL-fold/braceal/bba/model",
        params=params,
        config=tf_config,
    )

    print("estimator creation done")

    if params["mode"] == "train":
        print("train_steps:", params["train_steps"])
        est.train(input_fn=simulation_tf_record_input_fn, steps=params["train_steps"])

    weights_file = tf.train.latest_checkpoint(weights_dir)

    print(f"find weights done: {weights_file}")

    # gen = est.predict(
    #     input_fn=data_generator,
    #     checkpoint_path=weights_file,
    #     yield_single_examples=True,
    # )

    # print("predict done")

    # embeddings = np.array([list(it.values())[0] for it in gen])

    # print("embeddings gather done")

    # np.save(embeddings_file, embeddings)


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
