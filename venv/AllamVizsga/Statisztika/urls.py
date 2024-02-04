from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload', views.upload, name='upload'),
    path('arima', views.arima, name='arima'),
    path('download/', views.download, name='download_model'),

]
