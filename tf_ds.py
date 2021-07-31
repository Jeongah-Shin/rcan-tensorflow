import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


tfds.list_builders()[:10]

dataset, info = tfds.load(name='div2k', split=tfds.Split.TRAIN, with_info=True)

info.splits
