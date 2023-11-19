import smtpd
from typing import Any
from django.db import models
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
    
class Stat:
    megye_nev = ""
    atlag = szoras = varriancia = median = min = max = 0.0
    minDatum = maxDatum = ""
    adatok = []
    teszt_adatok = []
    teszt_idoszakok = []
    becslesek = []
    mse = 0
    rrmse = 0
    title = ""


    def __init__(self, megye_nev, adatok, idoPontok):
        self.megye_nev = megye_nev
        self.adatok = adatok
        self.atlag = round(np.mean(adatok), 2)
        self.szoras = round(np.std(adatok), 2)
        self.variancia = round(np.var(adatok), 2)
        self.median = round(np.median(adatok), 2)
        self.min = np.min(adatok)
        self.max = np.max(adatok)
        self.minDatum = idoPontok[list.index(adatok, self.min)]
        self.maxDatum = idoPontok[list.index(adatok, self.max)]
    

    def setTesztAdatok(self, teszt_adatok: list):
        self.teszt_adatok = teszt_adatok

    def MSE(self):
        n = len(self.teszt_adatok)
        return np.sum((self.teszt_adatok - self.becslesek)**2) / n 
    
    def RRMSE(self):
        mse = self.MSE()
        mean_y = np.mean(self.teszt_adatok)
        return np.sqrt(mse) / mean_y

        
    def AR(self, p: int, t:int):
        try:
            p = int(p)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, 0))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(t)
            self.becslesek = model_fit.forecast(t)
            return ([model_fit.summary(), forecast_values])
        
        except Exception as e:
            print(e)
    
    def MA(self, q: int, t:int):
        try:
            q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(0, 0, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(t)
            self.becslesek = model_fit.forecast(t)
            return ([model_fit.summary(), forecast_values])
        
        except Exception as e:
            print(e)
    
    def ARMA(self, p:int, q: int, t:int):
        try:
            p = int(p); q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(t)
            self.becslesek = model_fit.forecast(t)
            return ([model_fit.summary(), forecast_values])
        
        except Exception as e:
            print(e)
    
    def ARIMA(self, p:int, d: int, q: int, t:int):
        try:
            p = int(p); q = int(q); d = int(d)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, d, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(10)
            self.becslesek = model_fit.forecast(t)
            return ([model_fit.summary(), forecast_values])
   
        except Exception as e:
            print(e)
        
        
