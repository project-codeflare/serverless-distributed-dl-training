import tensorflow as tf
import horovod.tensorflow.keras as hvd

import ray
from horovod.ray import RayExecutor

import os
import tempfile
import gzip
import numpy as np
import ibm_boto3
from ibm_botocore.client import Config


def get_all_keys(cos, bucket, prefix='', suffix=''):
    kwargs = {'Bucket': bucket, 'Prefix': prefix}
    while True:
        resp = cos.list_objects_v2(**kwargs)
        try:
            contents = resp['Contents']
        except KeyError:
            return
        for obj in contents:
            key = obj['Key']
            if key.endswith(suffix):
                yield key
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break

def download(cos, bucket, prefix, local_dir):
    for key in get_all_keys(cos, bucket, prefix):
        local_file = os.path.join(local_dir, key[len(prefix)+1:])
        sub_dir = os.path.dirname(local_file)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        cos.download_file(bucket, key, local_file)

def upload(cos, local_dir, bucket, prefix):
    for path, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(path, file)
            key = os.path.join(prefix, local_file[len(local_dir)+1:])
            cos.upload_file(local_file, bucket, key)

def load_data(data_dir):
    files = [
      'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
      't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = [os.path.join(data_dir, file) for file in files]
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)
    return (x_train, y_train), (x_test, y_test)


def train(num_epochs, 
    api_key, instance_crn, endpoint_url, 
    data_bucket, data_prefix, 
    in_model_bucket, in_model_prefix, 
    out_model_bucket, out_model_prefix):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.rank()],'GPU')

    # Create a cos client
    cos = ibm_boto3.client("s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=instance_crn,
        endpoint_url=endpoint_url,
        config=Config(signature_version="oauth")
    )

    # Get input data from COS
    with tempfile.TemporaryDirectory() as tmpdir:
        download(cos, data_bucket, data_prefix, tmpdir)
        (x_train, y_train), (x_test, y_test) = load_data(tmpdir)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shard(hvd.size(), hvd.rank()).repeat().shuffle(10000).batch(128)

    
    # Get pre-trained model from COS to do transfer learning or fine-tuning or simply resume previous training
    with tempfile.TemporaryDirectory() as tmpdir:
        download(cos, in_model_bucket, in_model_prefix, tmpdir)
        model = tf.keras.models.load_model(tmpdir)

    '''
    Once you get the model, you can either do:
        a transfer learning by freezing model layers and only training on the last layer,
        or a fine-tuning by setting a very low learning rate and re-compile the model,
        or simply resume a previous training.
    For simplicity, here we just continue training the existing model with the given data.
    '''


    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(
            initial_lr=0.001 * hvd.size(), warmup_epochs=3, verbose=1),
    ]

    # Train the model.
    # Horovod: adjust number of steps based on number of GPUs.
    model.fit(
        dataset,
        steps_per_epoch=500 // hvd.size(),
        callbacks=callbacks,
        epochs=num_epochs,
        verbose=1 if hvd.rank() == 0 else 0)
    
    # Horovod: save model only from worker 0 to prevent other workers from corrupting it.
    if hvd.rank() == 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            upload(cos, tmpdir, out_model_bucket, out_model_prefix)

if __name__ == '__main__':
    ray.init(address="auto", _redis_password='5241590000000000')
    settings = RayExecutor.create_settings(timeout_s=30)
    executor = RayExecutor(settings, num_hosts=3, num_slots=2, use_gpu=False, cpus_per_slot=1)
    executor.start()
    executor.run(train, 
        kwargs=dict(
            num_epochs=2, 
            api_key='xxx', 
            instance_crn='xxx',
            endpoint_url='xxx', 
            data_bucket='horovod', 
            data_prefix='data/fashion', 
            in_model_bucket='horovod', 
            in_model_prefix='models/fashion/v1',
            out_model_bucket='horovod', 
            out_model_prefix='models/fashion/v2'))
    executor.shutdown()
