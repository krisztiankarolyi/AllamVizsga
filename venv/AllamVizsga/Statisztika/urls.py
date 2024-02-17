from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload', views.upload, name='upload'),
    path('arima', views.arima, name='arima'),
    path('Box-Jenkins', views.BoxJenkins, name="Box-Jenkins"),
    path('MLP', views.MLP, name="MLP"),
    path('MLPResults', views.MLPResults, name="MLPResults"),
]
