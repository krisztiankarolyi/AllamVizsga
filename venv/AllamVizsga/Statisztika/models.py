
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

class Stat :
    def __init__(self, megye_nev, adatok, idoPontok):
        self.megye_nev = megye_nev
        self.adatok = adatok
        self.idoszakok = idoPontok
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
        print(self.SeasonsAvg())

    
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

        self.adf["critical_values"] = {'5':0}
        self.adf["critical_values"]['5'] = round(adf_result[4]["5%"], 2)

        kpss_result = kpss(self.adatok)
        self.kpss["kpss_stat"] = round(kpss_result[0], 2)
        self.kpss["p_value"] = round(kpss_result[1], 2)

        self.kpss["critical_values"] = {'5':0}
        self.kpss["critical_values"]['5'] = round(kpss_result[3]["5%"], 2)


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

    def SeasonsAvg(self) -> dict:
        """returns a dictionary with total averages of the dataset for each season (winter, spring, summer, autumn)"""

        averages = {"winter": 0, "spring": 0, "summer": 0, "autumn": 0}
        winterCount = springCount = summerCount = AutumnCount = 0

        for i in range(len(self.adatok)):
            if "december" in self.idoszakok[i] or "január" in self.idoszakok[i] or "február" in self.idoszakok[i]:
                averages["winter"] += self.adatok[i]
                winterCount+=1
            elif "március" in self.idoszakok[i] or "április" in self.idoszakok[i] or "május" in self.idoszakok[i]:
                averages["spring"] += self.adatok[i]
                springCount+=1
            elif "június" in self.idoszakok[i] or "július" in self.idoszakok[i] or "augusztus" in self.idoszakok[i]:
                averages["summer"] += self.adatok[i]
                summerCount+=1
            elif "szeptember" in self.idoszakok[i] or "október" in self.idoszakok[i] or "november" in self.idoszakok[i]:
                averages["autumn"] += self.adatok[i]
                AutumnCount+=1

        averages["winter"] /= winterCount; averages["spring"] /= springCount; averages["summer"] /= summerCount; averages["autumn"] /= AutumnCount
        return averages
        
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

            self.becslesek = [round(value, 2) for value in model_fit.forecast(t)]
            self.aic = model_fit.aic

            return ([model_fit.summary(), self.becslesek])
        
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


    def predict_with_mlp(self, normalize=True, hidden_layers=(12, 12, 12), max_iters=3000, random_state=50):
        if not self.teszt_adatok:
            print("Nincsenek tesztelési adatok.")
            return
    
        X_train = np.arange(1, len(self.adatok) + 1).reshape(-1, 1)
        y_train = np.array(self.adatok)
        X_test = np.arange(len(self.adatok) + 1, len(self.adatok) + len(self.teszt_adatok) + 1).reshape(-1, 1)

        self.mlp_model = MLP(normalize, hidden_layers, max_iters, random_state)
        self.mlp_model.train_model(X_train, y_train)

        self.mlp_model.predictions = self.mlp_model.predict(X_test)

class MLP:
    def __init__(self, normalize=True, hidden_layers=(12, 12, 12), max_iters=3000, random_state=50):
        self.normalize = normalize
        self.hidden_layers = hidden_layers
        self.max_iters = max_iters
        self.random_state = random_state
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, max_iter=max_iters, random_state=random_state)
        self.scaler = StandardScaler()
        self.predictions = []

    def train_model(self, X_train, y_train):
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.normalize:
            X_test = self.scaler.transform(X_test)
        return self.model.predict(X_test)
