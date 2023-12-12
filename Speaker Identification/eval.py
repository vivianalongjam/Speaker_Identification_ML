import numpy as np
import librosa
import os
import pickle as pkl
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as Scaler
from sklearn.mixture import GaussianMixture as GMM


files = open('files.txt', 'r').read().split('\n')
audios = [librosa.load(f, sr=16000)[0] for f in files]
mfccs = [librosa.feature.mfcc(y=y,sr=16000,n_fft=400,hop_length=160,n_mfcc=20) for y in audios]
delta1 = [librosa.feature.delta(mfcc) for mfcc in mfccs]
delta2 = [librosa.feature.delta(mfcc, order=2) for mfcc in mfccs]

pca = pkl.load(open('pca.pkl', 'rb'))
models = pkl.load(open('models.pkl', 'rb'))
scaler = pkl.load(open('scaler.pkl', 'rb'))

mfccs = np.array(mfccs).reshape((len(mfccs), -1))
delta1 = np.array(delta1).reshape((len(delta1), -1))
delta2 = np.array(delta2).reshape((len(delta2), -1))

features = np.concatenate((delta1, mfccs, delta2), axis=1)
features = scaler.fit_transform(features)
features = pca.transform(features)
scores = np.zeros((len(features), len(models)))
for i, model in enumerate(models):
    scores[:, i] = model.score_samples(features)
preds = np.argmax(scores, axis=1)
for i, f in enumerate(files):
    print(f, preds[i])

