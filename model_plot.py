import os.path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import ast
from keras.models import load_model
import scipy.io as scio
import random
from tqdm import tqdm

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


def load_sca_model(model_file):
    try:
        model = load_model(model_file)
    except:
        print("Error: can't load Keras model file '%s'" % model_file)
        sys.exit(-1)
    return model


def get_prediction(model, Traces):
    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    if input_layer_shape[1] != len(Traces[0]):
        print("Error: model input shape %d instead of %d is not expected ..." % (
            input_layer_shape[1], len(Traces[0])))
        sys.exit(-1)

    # Adapt the data shape according our model input
    if len(input_layer_shape) == 2:
        # This is a MLP
        input_data = Traces
    elif len(input_layer_shape) == 3:
        # This is a CNN: reshape the data
        input_data = Traces
        input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
    else:
        print("Error: model input shape length %d is not expected ..." % len(input_layer_shape))
        sys.exit(-1)

    # Predict our probabilities
    predictions = model.predict(input_data)

    return predictions


def probability_cal(selected_Pts_interest, selected_predictions, NUMBER):
    probabilities_array = []

    for i in range(NUMBER):

        probabilities = np.zeros(256)

        for j in range(256):
            value = AES_Sbox[selected_Pts_interest[i] ^ j]

            probabilities[j] = selected_predictions[i][value]

        # print(probabilities)

        probabilities_array.append(probabilities)

        # print(probabilities_array)

    probabilities_array = np.array(probabilities_array)

    for i in range(len(probabilities_array)):
        if np.count_nonzero(probabilities_array[i]) != 256:
            none_zero_predictions = [a for a in probabilities_array[i] if a != 0]

            min_v = min(none_zero_predictions)

            #print(min_v**2)
            probabilities_array[i] = probabilities_array[i] + min_v**2

    return probabilities_array


def rank_cal(selected_probabilities, key_interest, NUMBER):
    rank = []

    total_pro = np.zeros(256)

    for i in range(NUMBER):
        # epsilon = 4*10**-12

        # selected_probabilities[i] = selected_probabilities[i] +epsilon

        total_pro = total_pro + np.log(selected_probabilities[i])

        # find the rank of real key in the total probabilities

        sorted_proba = np.array(list(map(lambda a: total_pro[a], total_pro.argsort()[::-1])))

        real_key_rank = np.where(sorted_proba == total_pro[key_interest])[0][0]

        rank.append(real_key_rank)

    rank = np.array(rank)

    return rank


if __name__ == "__main__":


    cut = 0
    stop = 10000
    # load traces and cut
    Traces = np.load('nor_traces_maxmin.npy')
    Traces = Traces[:, [i for i in range(130,240)]] #132,137 #130,240
    Traces = Traces[cut:stop]

    # load key
    key = np.load('key.npy')

    # load plaintext (all bytes)
    Pts = np.load('pt.npy')
    Pts = Pts[cut:stop]




    # choose interest key byte and pt byte
    interest_byte = 0
    key_interest = key[interest_byte]
    Pts_interest = Pts[:, interest_byte]

    # ==========================================================================

    # model path
    model_path = ''

    # Load model
    model = load_sca_model(model_path)

    # get predictions for all traces
    predictions = get_prediction(model, Traces)

    # randomly select trace for testing

    NUMBER = 10000


    average = 100

    ranks_array = []


    for i in tqdm(range(average), ncols=60):
        select = random.sample(range(len(Traces)), NUMBER)

        selected_Pts_interest = Pts_interest[select]
        selected_predictions = predictions[select]

        # calculate subkey probability for selected traces
        probabilities = probability_cal(selected_Pts_interest, selected_predictions, NUMBER)

        ranks = rank_cal(probabilities, key_interest, NUMBER)

        ranks_array.append(ranks)

    ranks_array = np.array(ranks_array)

    average_ranks = np.sum(ranks_array, axis=0) / average



    for i in range(len(average_ranks)):
        if average_ranks[i]<0.5:
            print(i)
            break
    np.save('average_ranks.npy',average_ranks)
    plt.plot(average_ranks)
    plt.show()


