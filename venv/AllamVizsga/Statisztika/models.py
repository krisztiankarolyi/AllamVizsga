import smtpd
from typing import Any
from django.db import models
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import base64
import io

class Stat:
    megye_nev = ""
    adf = {};
    kpss = {}
    atlag = szoras = varriancia = median = min = max = 0.0
    minDatum = maxDatum = ""
    adatok = []
    teszt_adatok = []
    teszt_idoszakok = []
    becslesek = []
    mse = 0; rrmse = 0
    title = ""
    pacf_acf_Diagram  = ""
    


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
        self.adf = {}; self.kpss = {}
        self.Stationarity()
    

    def setTesztAdatok(self, teszt_adatok: list):
        self.teszt_adatok = teszt_adatok

    def MSE(self):
        n = len(self.teszt_adatok)
        return np.sum((self.teszt_adatok - self.becslesek)**2) / n 
    
    def RRMSE(self):
        mse = self.MSE()
        mean_y = np.mean(self.teszt_adatok)
        return np.sqrt(mse) / mean_y
    
    def Stationarity(self):
        adf_result = adfuller(self.adatok)
        self.adf["adf_stat"] = adf_result[0]
        self.adf["p_value"] = adf_result[1]
        self.adf["critical_values"] = adf_result[4]

        kpss_result = kpss(self.adatok)
        self.kpss["kpss_stat"] = kpss_result[0]
        self.kpss["p_value"] = kpss_result[1]
        self.kpss["critical_values"] = kpss_result[3]

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
        
    def plot_acf_and_pacf(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
        fig.subplots_adjust(hspace=0.3)
        plot_acf(self.adatok, lags=40, ax=ax1, title=f"Autokorreláció ({self.megye_nev})")
        plot_pacf(self.adatok, lags=40, ax=ax2, title=f"Parciális Autokorreláció ({self.megye_nev})")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.pacf_acf_Diagram = encoded_image