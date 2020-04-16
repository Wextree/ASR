from django.shortcuts import render
from keras.models import load_model
from music.models import Music
import numpy as np
import librosa
import sklearn


# Create your views here.
from django.http import HttpResponse


def index(request):
    # return a start html
    return render(request, 'login.html')


# # get the wav path and create label and probability of it
def create_label(request):
    # get the address of the wav audio
    if request.method == 'POST':
        # wav_path = request.POST.get('path')
        file = request.FILES['file']
    # print(wav_path)

    wavName = "./music/wav/" + file.name
    print(wavName)
    with open(wavName, 'wb') as wav:
        for c in file.chunks():
            wav.write(c)


    music = Music()
    music.wav_path = wavName

    # load the web of ASR
    gen = load_model('music\MusicClass\music_cla_model.h5')

    # check the length of the audio and cut into a proper length
    x, sr = librosa.load(wavName)
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
    music.probability = max(probability)
    music.label = probability.index(max(probability))
    print("label：" + str(music.label))
    music.save()


    return HttpResponse("succeed!!!")

# def create_label(request):
#     # get the address of the wav audio
#     if request.method == 'POST':
#         # wav_path = request.POST.get('path')
#         file = request.FILES['file']
#     print(type(file))
#     print(file.name)
#     wavName = "./music/wav/" + file.name
#     print(wavName)
#     with open(wavName, 'wb') as wav:
#         for c in file.chunks():
#             wav.write(c)
#     return HttpResponse("succeed!!!")