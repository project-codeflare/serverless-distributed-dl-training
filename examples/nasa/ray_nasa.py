import tensorflow as tf
import horovod.tensorflow.keras as hvd

import ray
from horovod.ray import RayExecutor

import os
import time
import tempfile
import gzip
import numpy as np
import tarfile
import ibm_boto3
from ibm_botocore.client import Config

api_key = 'xxx'
instance_crn = 'xxx'
endpoint_url = 'xxx'


def upload(cos, local_dir, bucket, prefix):
    for path, _, files in os.walk(local_dir):
        for file in files:
            local_file = os.path.join(path, file)
            key = os.path.join(prefix, local_file[len(local_dir)+1:])
            cos.upload_file(local_file, bucket, key)

def gen_data(data, img_size, img_step, ts_size):
    height, width, ts_length = data.shape
    for row_start in range(0, height - img_size + 1, img_step):
        for col_start in range(0, width - img_size + 1, img_step):
            for ts_start in range(ts_length - ts_size):
                row_end = row_start + img_size
                col_end = col_start + img_size
                ts_end = ts_start + ts_size
                target_ts = ts_end
                x_train = data[row_start:row_end, col_start:col_end, ts_start:ts_end]
                y_train = data[row_start:row_end, col_start:col_end, target_ts:target_ts+1]
                yield x_train, y_train

def create_ds_low_mem(data):
    ds = tf.data.Dataset.from_generator(gen_data, 
                                    output_types=(tf.float32, tf.float32), 
                                    output_shapes=((32, 32, 3), (32, 32, 1)), 
                                    args=[data, 32, 16, 3])
    return ds

def create_ds_large_mem(data):
    img_size = 32
    img_step = 16
    ts_size = 3
    x_train = []
    y_train = []
    height, width, ts_length = data.shape
    for row_start in range(0, height - img_size + 1, img_step):
        for col_start in range(0, width - img_size + 1, img_step):
            for ts_start in range(ts_length - ts_size):
                row_end = row_start + img_size
                col_end = col_start + img_size
                ts_end = ts_start + ts_size
                target_ts = ts_end
                x_train.append(data[row_start:row_end, col_start:col_end, ts_start:ts_end])
                y_train.append(data[row_start:row_end, col_start:col_end, target_ts:target_ts+1])
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    return ds



def train():
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.rank()],'GPU')

    # Limit number of cpus per worker
    # tf.config.threading.set_intra_op_parallelism_threads(4)
    # tf.config.threading.set_inter_op_parallelism_threads(4)

    cos = ibm_boto3.client("s3",
        ibm_api_key_id=api_key,
        ibm_service_instance_id=instance_crn,
        endpoint_url=endpoint_url,
        config=Config(signature_version="oauth")
    )

    # Get input data from COS
    print('Loading data from COS')
    with tempfile.NamedTemporaryFile(suffix='.npy') as tmp:
        cos.download_file('horovod', 'data/nasa/MOD13Q1_250m_16_days_NDVI.npy', tmp.name)
        data = np.load(tmp.name)
    ds = create_ds_low_mem(data)
    # train_ds = ds.shard(hvd.size(), hvd.rank()).repeat().shuffle(10000).batch(128)
    train_ds = ds.shard(hvd.size(), hvd.rank()).repeat().batch(128)

    
    # Build model
    def downsample(filters, kernel_size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU()
        ])

    def upsample(filters, kernel_size):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def unet(input_channels):
        down_stack = [
            downsample(16, 3),  # (bs, 16, 16, 32)
            downsample(32, 3),  # (bs, 8, 8, 64)
            downsample(64, 3),  # (bs, 4, 4, 128)
            downsample(128, 3),  # (bs, 2, 2, 256)
            downsample(256, 3),  # (bs, 1, 1, 512)
        ]
        up_stack = [
            upsample(128, 3),  # (bs, 2, 2, 256 + 256)
            upsample(64, 3),  # (bs, 4, 4, 128 + 128)
            upsample(32, 3),  # (bs, 8, 8, 64 + 64)
            upsample(16, 3),  # (bs, 16, 16, 32 + 32)
        ]
        last = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same')  # (bs, 32, 32, 1)
        inputs = tf.keras.layers.Input(shape=[32, 32, input_channels])
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.concatenate([x, skip])
        outputs = last(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)

    model = unet(3)
    scaled_lr = 0.001 * hvd.size()
    opt = tf.optimizers.Adam(scaled_lr)
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)
    model.compile(opt,
        loss='mse',
        metrics=['mse'],
        experimental_run_tf_function=False)
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
    ]
    train_start = time.time()
    model.fit(train_ds, steps_per_epoch=5000 // hvd.size(), epochs=3, callbacks=callbacks, verbose=1 if hvd.rank() == 0 else 0)
    train_end = time.time()

    if hvd.rank() == 0:
        print ("Training time elapsed: ", train_end - train_start)

    # Horovod: save model only from worker 0 to prevent other workers from corrupting it.
    if hvd.rank() == 0:
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            upload(cos, tmpdir, 'horovod', 'models/nasa/v1')

if __name__ == '__main__':
    ray.init(address="auto", _redis_password='5241590000000000')
    settings = RayExecutor.create_settings(timeout_s=300)
    executor = RayExecutor(settings, num_hosts=4, num_slots=1, use_gpu=False, cpus_per_slot=4)
    executor.start()
    executor.run(train)
    executor.shutdown()
