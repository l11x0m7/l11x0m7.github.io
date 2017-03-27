--- 
layout: post 
title: Keras使用问题汇总
date: 2017-03-27 
categories: blog 
tags: [Deep Learning] 
description: Keras
--- 

# Keras使用问题汇总

## 1. Keras指定使用GPU

`CUDA_VISIBLE_DEVICES="0,1" python train.py`表示使用GPU0和GPU1。

## 2. Keras设定GPU使用内存大小(Tensorflow backend)

通过设定tensorflow的backend里的session实现。

```python
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

之后在运行代码中设置session：

```python
import keras.backend.tensorflow_backend as KTF
KTF.set_session(get_session())
```