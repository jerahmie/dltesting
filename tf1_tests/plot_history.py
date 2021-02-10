#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import h5py

def plot_history(model_file):
    """plot the training history from a a tensorflow project.
    """
    with h5py.File(model_file, 'r') as f:
        print("keys: %s" % f.keys())
    


if __name__ == "__main__":
    model_file = os.path.join('/mnt','Data','kaggle','dogs-vs-cats-small','save','cats_and_dogs_small_1.h5')
    plot_history(model_file)
