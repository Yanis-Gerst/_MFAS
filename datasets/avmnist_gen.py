# coding=utf-8
from tqdm import tqdm
"""
Created on May 28 2020.

This file is used to process the original mnist and generate the part of the image of the audio-visual mnist dataset.

@author slyviacassell
"""

import os
import re
import gzip
from multiprocessing import Pool, Lock, Manager

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import librosa
import scipy.io.wavfile as wav
from scipy import signal
import functools


def mnist_gen(root_path='./raw_data/mnist', img_saving_path='./avmnist/image', labels_saving_path='./avmnist/'):
    file_names = {'train_data': 'train-images-idx3-ubyte.gz', 'train_labels': 'train-labels-idx1-ubyte.gz',
                  'test_data': 't10k-images-idx3-ubyte.gz', 'test_labels': 't10k-labels-idx1-ubyte.gz'}

    working_dir = os.getcwd()
    print("Script working directory: %s" % working_dir)

    for key, file_name in file_names.items():
        file_path = os.path.join(root_path, file_name)
        print('file: %s' % key)
        with gzip.open(file_path, 'rb') as f:
            # read the definition of idx1-ubyte and idx3-ubyte
            f.seek(4)
            num = f.read(4)
            num = int().from_bytes(num, 'big')
            print('size of %s : %d' % (key, num))
            if re.match(r'.*data.*', key) is not None:
                height = f.read(4)
                height = int().from_bytes(height, 'big')
                width = f.read(4)
                width = int().from_bytes(width, 'big')

                data = np.frombuffer(f.read(), np.uint8).reshape(
                    num, height, width)

                # PCA projecting with 75% energy removing
                n_comp = int(height * width)
                pca = PCA(n_components=n_comp)
                projected = pca.fit_transform(
                    data.reshape(num, height * width))
                n_comp = (
                    (np.cumsum(pca.explained_variance_ratio_) > 0.25) != 0).argmax()
                rec = np.matmul(projected[:, :n_comp],
                                pca.components_[:n_comp])

                saved_path = os.path.join(working_dir, img_saving_path)
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                saved_name = key + '.npy'
                np.save(os.path.join(saved_path, saved_name), rec)
            else:
                data = np.frombuffer(f.read(), np.uint8)

                saved_path = os.path.join(working_dir, labels_saving_path)
                if not os.path.exists(saved_path):
                    os.makedirs(saved_path)
                saved_name = key + '.npy'
                np.save(os.path.join(saved_path, saved_name), data)


def wav_to_spectrogram(audio_dir, file_name, noise_path, f_length, t_length, noise_power, idx_cnt, output):
    """ Creates a spectrogram of a wav file.

    :param audio_dir: path of wav files
    :param file_name: file name of the wav file to process
    :param noise_path: path of noise wav file
    :return:
    """
    # if use librosa to process the wav file, the value of each element would be very small
    audio_path = os.path.join(audio_dir, file_name)

    y, sr = librosa.load(audio_path, sr=None)

    with lck:
        # choose one channel of greys
        output['data'].append(y.tolist())
        idx_cnt.value += 1

    return y


def pool_init(l):
    global lck
    lck = l


def dir_to_spectrogram(audio_dir, saving_dir, noise_dir, labels_dir, num_processes, f_length, t_length, noise_power):
    """ Creates spectrograms of all the audio files in a dir

    :param audio_dir: path of directory with audio files
    :param noise_dir: path to nosie audio files
    :return:
    """

    m = Manager()
    l = m.Lock()
    cnt = m.Value('int', 0)
    audio_spectrogram = m.dict({'data': m.list()})

    wav_dir = os.path.join(audio_dir, 'recordings')
    file_names = [f for f in os.listdir(wav_dir) if os.path.isfile(
        os.path.join(wav_dir, f)) and '.wav' in f]

    if len(file_names) == 0:
        print('No .wav file in %s' % wav_dir)
        exit(1)

    # set the size of the spectrogarm to (112, 112)
    plt.rcParams['figure.figsize'] = [1.12, 1.12]
    plt.rcParams['figure.dpi'] = 100
    noise_names = get_noise_names(noise_dir)

    # # 4 speakers and 50 wav files of each digit per speaker
    # speakers = ['jackson', 'nicolas', 'theo', 'yweweler']
    # test_speaker = speakers[-1]
    #
    # # 50 noise files
    # train_noise_names = noise_names[:40]
    # test_noise_names = noise_names[-10:]
    #
    # train_category = {str(i): list() for i in range(10)}
    # test_category = {str(i): list() for i in range(10)}
    # for file_name in file_names:
    #     if test_speaker in file_name:
    #         test_category[file_name[0]].append(file_name)
    #     else:
    #         train_category[file_name[0]].append(file_name)
    #
    # train_labels = np.load(os.path.join(labels_dir, 'train_labels.npy'))
    # test_labels = np.load(os.path.join(labels_dir, 'test_labels.npy'))
    #
    # train_names = []
    # train_noises = []
    # test_names = []
    # test_noises = []
    #
    # idx_list = [0 for i in range(10)]
    # noise_idx = 0
    # for train_label in train_labels:
    #     train_names.append(train_category[str(train_label)][idx_list[train_label]])
    #     idx_list[train_label] += 1
    #     idx_list[train_label] %= 150
    #
    #     train_noises.append(train_noise_names[noise_idx])
    #     noise_idx += 1
    #     noise_idx %= 40
    #
    # idx_list = [0 for i in range(10)]
    # noise_idx = 0
    # for test_label in test_labels:
    #     test_names.append(test_category[str(test_label)][idx_list[test_label]])
    #     idx_list[test_label] += 1
    #     idx_list[test_label] %= 50
    #
    #     test_noises.append(test_noise_names[noise_idx])
    #     noise_idx += 1
    #     noise_idx %= 10
    #
    # names = train_names + test_names
    # noises = train_noises + test_noises

    # 50 noise files
    train_noise_names = noise_names[:40]
    test_noise_names = noise_names[-10:]

    train_category = {str(i): list() for i in range(10)}
    test_category = {str(i): list() for i in range(10)}

    test_wav_idx = [i for i in range(40, 50)]
    for file_name in file_names:
        idx = file_name.rfind('_') + 1
        if int(file_name[idx:-4]) not in test_wav_idx:
            train_category[file_name[0]].append(file_name)
        else:
            test_category[file_name[0]].append(file_name)

    train_labels = np.load(os.path.join(labels_dir, 'train_labels.npy'))
    test_labels = np.load(os.path.join(labels_dir, 'test_labels.npy'))
    train_names = []
    train_noises = []
    test_names = []
    test_noises = []

    idx_list = [0 for i in range(10)]
    noise_idx = 0
    for train_label in train_labels:
        train_names.append(
            train_category[str(train_label)][idx_list[train_label]])
        idx_list[train_label] += 1
        idx_list[train_label] %= 160

        train_noises.append(train_noise_names[noise_idx])
        noise_idx += 1
        noise_idx %= 40

    idx_list = [0 for i in range(10)]
    noise_idx = 0
    for test_label in test_labels:
        test_names.append(test_category[str(test_label)][idx_list[test_label]])
        idx_list[test_label] += 1
        idx_list[test_label] %= 40

        test_noises.append(test_noise_names[noise_idx])
        noise_idx += 1
        noise_idx %= 10

    names = train_names + test_names
    noises = train_noises + test_noises

    print('start')
    pool = Pool(processes=num_processes, initializer=pool_init, initargs=(l,))
    for i in tqdm(range(len(names))):
        pool.apply_async(wav_to_spectrogram,
                         args=(wav_dir, names[i], noises[i], f_length, t_length, noise_power,
                               cnt, audio_spectrogram))
    # for i in tqdm(range(len(names))):
    #     wav_to_spectrogram(wav_dir, names[i], noises[i], f_length, t_length, noise_power,
    #                        cnt, audio_spectrogram)

    # close the pool and reject all the new process request
    pool.close()
    # waiting until all the tasks are finished
    print("pool waiting")
    pool.join()

    max_len = max(len(wave) for wave in audio_spectrogram["data"])
    padded_waves = [pad_wave(wave, max_len)
                    for wave in audio_spectrogram["data"]]
    print(max_len)
    data = np.array(padded_waves)

    # data = np.array([i for i in audio_spectrogram['data']])
    # labels = np.array([i for i in audio_spectrogram['labels']])

    train_data = data[:60000]
    test_data = data[60000:]
    print(data.shape)
    np.save(os.path.join(saving_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(saving_dir, 'test_data.npy'), test_data)


def get_noise_names(noise_dir):
    csv = pd.read_csv(os.path.join(noise_dir, 'meta/esc50.csv'))

    # sample one recording from each category
    recordings = csv.groupby('target')['filename'].apply(
        lambda cat: cat.sample(1)).reset_index()['filename']
    file_names = recordings.tolist()
    file_names = [os.path.join(noise_dir, 'audio', i) for i in file_names]

    return file_names


def audio_gen(audio_dir='./raw_data/FSDD/', saving_dir='./avmnist/audio',
              noise_dir='./raw_data/ESC-50/', noise_power=0, labels_dir='./avmnist/'):
    working_dir = os.getcwd()
    # note that the size of the spectrogram array is not (112, 112),
    # but the spectrum size is (112, 112).
    # read the doc of plt.specgram()
    dir_to_spectrogram(os.path.join(working_dir, audio_dir), os.path.join(working_dir, saving_dir),
                       os.path.join(working_dir, noise_dir), labels_dir, f_length=112, t_length=112, num_processes=32,
                       noise_power=noise_power)


def pad_wave(wave, target_len):
    return np.pad(wave, (0, target_len - len(wave)), mode='constant')


if __name__ == '__main__':
    # mnist_gen()
    print("Audio gen...")
    audio_gen()

    print('ok')
