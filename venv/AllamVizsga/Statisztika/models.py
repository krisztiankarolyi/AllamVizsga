import smtpd
from typing import Any
from django.db import models
import numpy as np
import statsmodels.api as sm
    
class Stat:
    megye_nev = ""
    atlag = szoras = varriancia = median = min = max = 0.0
    minDatum = maxDatum = ""
    adatok = []
    def __init__(self, megye_nev, adatok, idoPontok):
        self.megye_nev = megye_nev
        self.adatok = adatok
        self.atlag = round(np.mean(adatok), 2)
        self.szoras = round(np.std(adatok), 2)
        self.variancia = round(np.var(adatok), 2)
        self.median = round(np.median(adatok), 2)
        self.min = np.min(adatok)
        self.max = np.max(adatok)
        minIndex = list.index(adatok, self.min)
        maxIndex = list.index(adatok, self.max)
        self.minDatum = idoPontok[minIndex]
        self.maxDatum = idoPontok[maxIndex]
    
    def AR(self, p: int):
        try:
            p = int(p)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, 0))
            model_fit = model.fit()
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def MA(self, q: int):
        try:
            q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(0, 0, q))
            model_fit = model.fit()
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def ARMA(self, p:int, q: int):
        try:
            p = int(p); q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, q))
            model_fit = model.fit()
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def ARIMA(self, p:int, d: int, q: int):
        try:
            p = int(p); q = int(q); d = int(d)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, d, q))
            model_fit = model.fit()
            return (model_fit.summary())
        
        except Exception as e:
            print(e)