from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('mus/', views.create_label, name='check'),
]