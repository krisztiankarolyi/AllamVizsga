
import traceback
from typing import Any
from django.db import models
from matplotlib import pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import base64
import json
import io

class Stat :
    megye_nev = ""
    adf = {}
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
        self.aic = 0
        self.Stationarity()
    

    def setTesztAdatok(self, teszt_adatok: list):
        self.teszt_adatok = teszt_adatok


    def MSE(self):
        try:
            n = len(self.teszt_adatok)
            teszt_adatok_np = np.array(self.teszt_adatok)
            becslesek_np = np.array(self.becslesek)
            
            self.mse = np.sum((teszt_adatok_np - becslesek_np)**2) / n
            return self.mse
        except:
            return -1   
    
    def RRMSE(self):
        try:
            mse = self.MSE()
            mean_y = np.mean(self.teszt_adatok)
            
            # Ellenőrizze, hogy a gyök alatt lévő kifejezés értéke negatív-e
            if mse < 0 or mean_y <= 0:
                self.rrmse = np.sqrt(-1*(mse)) / mean_y
            else:  
                self.rrmse = np.sqrt(mse) / mean_y
            return self.rrmse
        
        except Exception as e:
            print(e)
            return -1

    
    def Stationarity(self):
        adf_result = adfuller(self.adatok)
        self.adf["adf_stat"] = round(adf_result[0], 2)
        self.adf["p_value"] = round(adf_result[1], 2)
        self.adf["critical_values"] = {1: 0, 5:0, 10: 0}

        self.adf["critical_values"]['1'] = round(adf_result[4]["1%"], 2)
        self.adf["critical_values"]['5'] = round(adf_result[4]["5%"], 2)
        self.adf["critical_values"]['10'] = round(adf_result[4]["10%"], 2)

        kpss_result = kpss(self.adatok)
        self.kpss["kpss_stat"] = round(kpss_result[0], 2)
        self.kpss["p_value"] = round(kpss_result[1], 2)

        self.kpss["critical_values"] = {1: 0, 5:0, 10: 0}
        self.kpss["critical_values"]['1'] = round( kpss_result[3]["1%"],2)
        self.kpss["critical_values"]['5'] = round(kpss_result[3]["5%"], 2)
        self.kpss["critical_values"]['10'] = round(kpss_result[3]["10%"], 2)

    def AR(self, p: int, t:int):
        try:
            p = int(p)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, 0))
            model_fit = model.fit()
            self.becslesek = model_fit.forecast(t)
            self.aic= model_fit.aic

            return ([model_fit.summary(), self.becslesek])
        
        except Exception as e:
            print(e)
    
    def MA(self, q: int, t:int):
        try:
            q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(0, 0, q))
            model_fit = model.fit()
            self.becslesek = model_fit.forecast(t)
            self.aic= model_fit.aic

            return ([model_fit.summary(), self.becslesek])
        
        except Exception as e:
            print(e)
    
    def ARMA(self, p:int, q: int, t:int):
        try:
            p = int(p); q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, q))
            model_fit = model.fit()
            
            # Módosítás: Csak a tesztadatok méretéig számoljuk ki a becsléseket
            self.becslesek = [round(value, 2) for value in model_fit.forecast(t)]
            self.aic = model_fit.aic

            return ([model_fit.summary(), self.becslesek])
        
        except Exception as e:
            print(e)
        
        except Exception as e:
            print(e)
    
    def ARIMA(self, p:int, d: int, q: int, t:int):
        try:
            p = int(p); q = int(q); d = int(d)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, d, q))
            model_fit = model.fit()
            self.becslesek = model_fit.forecast(t)
            self.aic= model_fit.aic

            return ([model_fit.summary(), self.becslesek])
   
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

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(**data)

class StatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Stat):
            return obj.__dict__
        return super().default(obj)
