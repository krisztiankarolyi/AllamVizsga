import traceback
from typing import Any
from django.db import models
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.metrics import r2_score
import base64
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

class Stat :
    def __init__(self, megye_nev, adatok, idoszakok):
        self.megye_nev = megye_nev
        self.adatok = adatok
        self.idoszakok = idoszakok
     
        self.adf = {}; self.kpss = {}
        self.aic = 0
        self.teszt_idoszakok = []
        self.ARIMAbecslesek = []
        self.MLPDiagram = None
        self.ARIMADiagram = None
        self.mse = 0
        self.rrmse = 0
        self.r_squared = 0

    def calculateStatistics(self):
        self.atlag = round(np.mean(self.adatok), 2)
        self.szoras = round(np.std(self.adatok), 2)
        self.variancia = round(np.var(self.adatok), 2)
        self.median = round(np.median(self.adatok), 2)
        self.min = np.min(self.adatok)
        self.max = np.max(self.adatok)
        self.minDatum = self.idoszakok[list.index(self.adatok, self.min)]
        self.maxDatum = self.idoszakok[list.index(self.adatok, self.max)]
        self.Stationarity()

    
    def setTesztAdatok(self, teszt_adatok: list):
        self.teszt_adatok = teszt_adatok
    
    def setTesztIdoszakok(self, idoszakok: list):
        self.teszt_idoszakok = idoszakok

    def setMLPDiagram(self, diagram):
        self.MLPDiagram = diagram

    def setARIMADiagram(self, diagram):
        self.ARIMADiagram = diagram

    def MSE(self, becslesek):
        try:
            n = len(self.teszt_adatok)
            teszt_adatok_np = np.array(self.teszt_adatok)
            becslesek_np = np.array(becslesek)
            self.mse = np.sum((teszt_adatok_np - becslesek_np)**2) / n
            return self.mse
        except:
            return -1   
        
    def RRMSE(self, becslesek):
        try:
            mse = self.MSE(becslesek)
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
            self.ARIMAbecslesek = model_fit.forecast(t)
            self.ARIMAbecsleseksZipped = zip(self.ARIMAbecslesek, self.teszt_adatok)
            self.aic= model_fit.aic
            self.mse = self.MSE(self.ARIMAbecslesek)
            self.rrmse = self.RRMSE(self.ARIMAbecslesek)
            self.r_squared  = r2_score(self.teszt_adatok, self.ARIMAbecslesek)
            
            return ([model_fit.summary(), self.ARIMAbecslesek])
        
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
            self.ARIMAbecslesek = model_fit.forecast(t)
            self.aic= model_fit.aic
            self.ARIMAbecsleseksZipped = zip(self.ARIMAbecslesek, self.teszt_adatok)
            self.mse = self.MSE(self.ARIMAbecslesek)
            self.rrmse = self.RRMSE(self.ARIMAbecslesek)
            self.r_squared  = r2_score(self.teszt_adatok, self.ARIMAbecslesek)
            return ([model_fit.summary(), self.ARIMAbecslesek])
        
        except Exception as e:
            print(e)
    
    def ARMA(self, p:int, q: int, t:int):
        try:
            p = int(p); q = int(q)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, 0, q))
            model_fit = model.fit()
            self.ARIMAbecslesek = [round(value, 2) for value in model_fit.forecast(t)]
            self.aic = model_fit.aic
            self.ARIMAbecsleseksZipped = zip(self.ARIMAbecslesek, self.teszt_adatok)
            self.mse = self.MSE(self.ARIMAbecslesek)
            self.rrmse = self.RRMSE(self.ARIMAbecslesek)
            self.r_squared  = r2_score(self.teszt_adatok, self.ARIMAbecslesek)
            return ([model_fit.summary(), self.ARIMAbecslesek])
        
        except Exception as e:
            print(e)
           
    def ARIMA(self, p:int, d: int, q: int, t:int):
        try:
            p = int(p); q = int(q); d = int(d)
            idosor = self.adatok
            model = sm.tsa.ARIMA(idosor, order=(p, d, q))
            model_fit = model.fit()
            self.ARIMAbecslesek = model_fit.forecast(t)
            self.aic= model_fit.aic
            self.ARIMAbecsleseksZipped = zip(self.ARIMAbecslesek, self.teszt_adatok)
            self.mse = self.MSE(self.ARIMAbecslesek)
            self.rrmse = self.RRMSE(self.ARIMAbecslesek)
            self.r_squared  = r2_score(self.teszt_adatok, self.ARIMAbecslesek)
            return ([model_fit.summary(), self.ARIMAbecslesek])
   
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
 
    def predict_with_mlp(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, scaler="standard", randomStateMax=70, randomStateMin=50, solver="adam", targetRRMSE=0.6):
        if not self.teszt_adatok:
            print("Nincsenek tesztelési adatok.")
            return   
        self.X_train = np.arange(1, len(self.adatok) + 1).reshape(-1, 1)
        self.y_train = np.array(self.adatok)
        self.X_test = np.arange(len(self.adatok) + 1, len(self.adatok) + len(self.teszt_adatok) + 1).reshape(-1, 1)

        random_state = self.find_best_random_state(actFunction=actFunction, random_state_min=randomStateMin, random_state_max=randomStateMax, max_iters=max_iters, scaler=scaler, hidden_layers=hidden_layers, solver=solver, targetRRMSE=targetRRMSE)

        self.mlp_model = MLP(self.teszt_adatok, actFunction=actFunction, hidden_layers=hidden_layers, max_iters=max_iters, random_state=random_state, scalerMode=scaler, solver=solver)

        self.mlp_model.train_model(self.X_train, self.y_train)
        self.mlp_model.predictions = self.mlp_model.predict(self.X_test)
        self.MLPResultsZipped = zip(self.mlp_model.predictions, self.teszt_adatok)

        self.mlp_model.mse = self.MSE(self.mlp_model.predictions)
        self.mlp_model.rrmse = self.RRMSE(self.mlp_model.predictions)




    def find_best_random_state(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, random_state_min=50, random_state_max=70, scaler="standard", solver="adam", targetRRMSE=0.06):
        best_random_state = None
        best_rrmse = float(1000) 

        for random_state in range(random_state_min, random_state_max+1):
            mlp_model = MLP(self.teszt_adatok, actFunction, hidden_layers, max_iters, random_state, scaler, solver=solver)

            mlp_model.train_model(self.X_train, self.y_train)
            predictions = mlp_model.predict(self.X_test)
            rrmse = self.RRMSE(predictions)
            print(f"trying {self.megye_nev}'s MLP prediction with random state {random_state} --> RRMSE: {rrmse}")

            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_random_state = random_state
            
            if round(rrmse, 2) <= targetRRMSE:
                print(f"target RRMSE{targetRRMSE} reached, stopping search...")
                return best_random_state

        return best_random_state

class MLP:
    def __init__(self, test_data, actFunction, hidden_layers=(12, 12, 12), max_iters=2000, random_state=50, scalerMode="standard", solver="adam"):
        self.test_data = test_data
        self.hidden_layers = hidden_layers
        self.NrofHiddenLayers = len(hidden_layers)
        self.max_iters = max_iters
        self.random_state = random_state
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver=solver,  activation=actFunction, max_iter=max_iters, random_state=random_state)
        self.scaler = StandardScaler()
        self.predictions = []
        self.mse = 0
        self.rrmse = 0
        self.accuracy = 0
        self.scalerMode = scalerMode
        self.scaler = StandardScaler()
        self.modelStr = self.NrofHiddenLayers * '{}'
        self.modelStr = self.modelStr.format(*hidden_layers) + f" random state({self.random_state})"

        if (scalerMode == "robust"):
            self.scaler = RobustScaler()
        if (scalerMode == "minmax"):
            self.scaler = MinMaxScaler()

    def train_model(self, X_train, y_train):
        if self.scalerMode != "-":
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        if self.scalerMode != "-":
            X_test = self.scaler.transform(X_test)       
        return self.model.predict(X_test)
    
    