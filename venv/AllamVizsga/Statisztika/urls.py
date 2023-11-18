from django.urls import path
from . import views

urlpatterns = [
    path('/', views.home, name='home'),
    path('/Statistics', views.statistics, name='statistics'),
    path('/arima', views.arima, name='arima'),
    path('/arimaForecasts', views.arimaForecasts, name='arimaForecasts'),
]
