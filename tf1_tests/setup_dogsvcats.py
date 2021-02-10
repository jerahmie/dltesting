#!/usr/bin/env python
""" Convolutional neural net for dogs vs cats trainging.
    Following the examples from Chapter 5 in 
    "Deep Learning with Python" by Francois Chollet.
"""

import os
import shutil

#original_dataset_dir = os.path.join('/mnt','Data','kaggle','dogs-vs-cats','train')
#base_dir = os.path.join('/mnt','Data','kaggle','dogs-vs-cats-small')
file_path=os.path.dirname(os.path.abspath(__file__))
original_dataset_dir = os.path.join(file_path,'dogs_vs_cats','train')
base_dir = os.path.join(file_path, 'dogs_vs_cats_small')

try:
    os.mkdir(base_dir)
except FileExistsError:
    print('base_dir exists: ', base_dir)

train_dir = os.path.join(base_dir, 'train')
try:
    os.mkdir(train_dir)
except FileExistsError:
    print('train_dir exists: ', train_dir)

validation_dir = os.path.join(base_dir, 'validation')
try:
    os.mkdir(validation_dir)
except FileExistsError:
    print('validation_dir exists: ', validation_dir)

test_dir = os.path.join(base_dir, 'test')
try:
    os.mkdir(test_dir)
except FileExistsError:
    print('test_dir exists: ', test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
try:
    os.mkdir(train_cats_dir)
except FileExistsError:
    print('train_cats_dir exists: ', train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
try:
    os.mkdir(train_dogs_dir)
except FileExistsError:
    print('train_dogs_dir exists: ', train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
try:
    os.mkdir(validation_cats_dir)
except:
    print('validation_cats_dir exists: ', validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
try:
    os.mkdir(validation_dogs_dir)
except:
    print('validation_dogs_dir exists: ', validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
try:
    os.mkdir(test_cats_dir)
except:
    print('test_cats_dir exists: ', test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
try:
    os.mkdir(test_dogs_dir)
except:
    print('test_dogs_dir exists: ', test_dogs_dir)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

