# Serverless Distributed Deep Learning Training with Code Engine

This tutorial walks through the steps to train a deep learning model in a ***serverless*** and ***distributed*** environment with IBM Code Engine.

## Components

- **Code Engine**: provides the underlying serverless environment.
- **Ray**: provides a Ray cluster on Code Engine for easily running distributed applications.
- **Horovod**: provides a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.

The **serverless** and **distributed** deep learning training is achieved by **running Horovod code on a Ray cluster which launched in Code Engine**.

## Steps

1. Setup Code Engine. 
    1. Create a [Code Engine](https://cloud.ibm.com/catalog/services/codeengine) project in IBM Cloud.
    2. Enable the project with Ray. Currently this can only be done by sharing the namespace id of your project with Code Engine team and ask their help to enable it. **Update: now you can do it yourself: https://www.ibm.com/cloud/blog/ray-on-ibm-cloud-code-engine**
2. Prepare Ray Cluster file. Ray cluster is configured and provisioned by a single ray cluster yaml file. All cluster-related inforation (nodes spec, container image, resouce allocation, etc.) should be configured in this file. Here is a [template](examples/mnist/ray_cluster.yaml).
3. Prepare training script. Modify your existing training script to use Horovod on Ray.
    1. Modify for Horovod. The modification varies a little bit by your favorite framework, but is very easy and straightforward. You can find more information on [Horovod with Tensorflow](https://horovod.readthedocs.io/en/stable/tensorflow.html), [Horovod with Tensorflow Keras](https://horovod.readthedocs.io/en/stable/keras.html), [Horovod with PyTorch](https://horovod.readthedocs.io/en/stable/pytorch.html) and [Horovod on MXNet](https://horovod.readthedocs.io/en/stable/mxnet.html).
    2. Modify for Ray integration. Script for running on Horovod-Ray is slightly different from running on Horovod alone, and a few more modifications for ray integration is required. This is as simple as wrapping your code from previous step in a train() function and execute it with RayExecutor. You can find the example [here](examples/mnist/ray_mnist.py)
4. Run it! Running is as simple as a two-line script:
    ```bash
    # Launch Ray cluster
    ray up -y --no-config-cache ray_cluster.yaml 
    # Submit your training
    ray submit --no-config-cache ray_cluster.yaml ray_mnist.py
    ```
    You might have to wait couple minutes for the cluster launch to complete after running the first command, if you are launching it for the first time (as it takes time to pull the image and allocate the resources).
    
## Examples
You can find examples [here](examples).  
[mnist](examples/mnist) contains an example of training Fashion MNIST model where data is consumed from COS and trained model is saved back to COS.  
[mnist-transfer-learning](examples/mnist-transfer-learning) contains an example of doing transfer learning or fine-tuning or reusume-training where data is consumed from COS, pre-trained model is loaded from COS and trained model is saved back to COS.

To run the demo (e.g. mnist), just do:
```bash
# 1. Connect to your code engine project, e.g.
export KUBECONFIG=/Users/lchu/.bluemix/plugins/code-engine/horovod-5dc3ff50-23e7-46be-92b2-e2de2da9bd71.yaml
# 2. Launch Ray cluster
ray up -y --no-config-cache ray_cluster.yaml 
# 3. Submit your training
cd examples/mnist
### Modify information as needed, e.g. change credentials in ray_mnist.py, change resouce allocation in ray_cluster.yaml, etc. ###
ray submit --no-config-cache ray_cluster.yaml ray_mnist.py
```
