from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('home', views.home, name='home'),
    path('arima', views.arima, name='arima'),
    path('Box-Jenkins', views.BoxJenkins, name="Box-Jenkins"),
    path('MLP', views.MLP, name="MLP"),
    path('MLP_Results', views.MLPResults, name="MLPResults"),
    path('LSTM', views.LSTM, name="LSTM"),
    path('LSTM_Results', views.LSTMResults, name="LSTMResults"),
]
