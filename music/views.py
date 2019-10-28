from django.shortcuts import render
from keras.models import load_model
import numpy as np
import librosa
import sklearn


# Create your views here.
from django.http import HttpResponse


def index(request):
    # return a start html
    return render(request, 'login.html')


# get the wav path and create label and probability of it
def create_label(request):
    # get the address of the wav audio
    if request.method == 'POST':
        wav_path = request.POST.get('path')
    print(wav_path)

    # load the web of ASR
    gen = load_model('music\MusicClass\music_cla_model.h5')

    # check the length of the audio and cut into a proper length
    x, sr = librosa.load(wav_path)
    if len(x) < 88200:
        x = np.pad(x, (0, 88200 - x.shape[0]), 'constant')
    else:
        x = x[0:88200:1]

    # add some feature and reshape it
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    norm_mfcc = sklearn.preprocessing.scale(mfcc, axis=1)
    mfcc = norm_mfcc.reshape(1, mfcc.shape[0], mfcc.shape[1], 1)

    # predict the result and get the best one and its label
    result = gen.predict(mfcc)
    probability = []
    result = result.reshape(10, 1)
    for num in result:
        probability.append(float(num))
    print(probability, sum(probability), max(probability))


    return HttpResponse("succeed!!!")