from django.db import models

# Create your models here.
class Music(models.Model):
    # the address of audio
    wav_path = models.CharField(max_length=200)
    # the feature of the audio(the best one)
    label = models.CharField(max_length=20)
    # the probability of the label
    probability = models.FloatField()