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
    
    def showForecast(self, forecast_values, tipus):
        plt.close("all")
        
        forecast_index = ["2022 október", "2022 november", "2022 december", "2023 január", "2023 február", "2023 március", "2023 április", "2023 május", "2023 június", "2023 július"]
        plt.plot(forecast_index, forecast_values, label='Előrejelzések - ', color='red')
        plt.title("Előrejelzés " + self.megye_nev + tipus)

        # Állítsuk be az Y tengely skáláját
     #   plt.ylim(2, 5.5)
      #  plt.yticks(np.arange(2, 5.6, 0.5))  # Y tengely léptéke 0.5 lépésenként

        # Hozzáadunk függőleges segédvonalakat
       # for y in np.arange(2, 5.6, 0.5):
        #    plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)

        print(self.megye_nev + " " + tipus + ":\n")
        for i in forecast_values:
            print(str(i) + "\n")
        #plt.legend()
        #plt.show()
        
    def AR(self, p: int):
        try:
            p = int(p)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, 0))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(10)
            self.showForecast(forecast_values, "AR("+str(p)+")")
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def MA(self, q: int):
        try:
            q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(0, 0, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(10)
            self.showForecast(forecast_values, "MA("+str(q)+")")
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def ARMA(self, p:int, q: int):
        try:
            p = int(p); q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(10)
            self.showForecast(forecast_values, "ARMA("+str(p)+", "+str(q)+")")
            return (model_fit.summary())
        
        except Exception as e:
            print(e)
    
    def ARIMA(self, p:int, d: int, q: int):
        try:
            p = int(p); q = int(q); d = int(d)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, d, q))
            model_fit = model.fit()
            forecast_values = model_fit.forecast(10)
            self.showForecast(forecast_values, "ARMA("+str(p)+", "+str(d)+", "+(q)+")")
            return (model_fit.summary())
   
        except Exception as e:
            print(e)
        
        
