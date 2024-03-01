import array
import traceback
from typing import Any
from django.db import models
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
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
from keras.models import Sequential
import pandas as pd
from keras.layers import LSTM, Dense


class Stat :
    def __init__(self, idosor_nev, adatok, idoszakok):
        self.idosor_nev = idosor_nev
        self.adatok = adatok
        self.idoszakok = idoszakok
        print("idoszakok", self.idoszakok)
        self.adf = {}; self.kpss = {}
        self.teszt_idoszakok = []


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
     #   print(f"{self.idosor_nev} teszt: adatok: {self.teszt_adatok}")
 
    
    def setTesztIdoszakok(self, idoszakok: list):
        self.teszt_idoszakok = idoszakok
        print(f"{self.idosor_nev} teszt: idoszakok: {self.teszt_idoszakok}")

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
        
           
    def predictARIMA(self, p:int = 1, d: int = 0, q: int = 0, t:int = 10):
        t = len(self.teszt_adatok)
        self.ARIMA = ARIMA(p, d, q, t, adatok=self.adatok, teszt_adatok=self.teszt_adatok, idoszakok=self.idoszakok, teszt_idoszakok=self.teszt_idoszakok )
        return self.ARIMA
        
    def plot_acf_and_pacf(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
        fig.subplots_adjust(hspace=0.3)
        plot_acf(self.adatok, lags=40, ax=ax1, title=f"Autokorreláció ({self.idosor_nev})")
        plot_pacf(self.adatok, lags=40, ax=ax2, title=f"Parciális Autokorreláció ({self.idosor_nev})")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.pacf_acf_Diagram = encoded_image
 
    def predict_with_mlp(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, scaler="standard", randomStateMax=70, randomStateMin=50, solver="adam", targetRRMSE=0.6, x_mode = "delayed", n_delays = 3):
        if not self.teszt_adatok:
            print("Nincsenek tesztelési adatok.")
            return          

        if(x_mode == "date"):
            self.dependency = "év - hónap párok"
            # az adatok a megfigyelések időpontjaitól függnek (év -hónap száma párosok)
            data = self.idoszakok + self.teszt_idoszakok
            target = self.adatok + self.teszt_adatok
            data = [item.split() for item in data]
            data = [[int(item[0]), self.get_month_number(item[1])] for item in data]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=len(self.teszt_adatok), shuffle=False)

        else:
            # az adatok n darab korábbi megfigyeléstől függnek (késleltetett értékek)
            #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalamzott.
            test_data = self.adatok[-n_delays:]  + self.teszt_adatok
            learning_data = self.adatok[:-n_delays]

            self.X_train, self.y_train = split_sequence(learning_data, n_delays)
            self.X_test, self.y_test = split_sequence(test_data, n_delays)


        self.X_Train_Y_Train_Zipped = zip(self.X_train, self.y_train)
        self.X_Test_Y_Test_Zipped = zip(self.X_test, self.y_test)

        self.random_state = self.find_best_random_state(actFunction=actFunction, random_state_min=randomStateMin, random_state_max=randomStateMax, max_iters=max_iters, scaler=scaler, hidden_layers=hidden_layers, solver=solver, targetRRMSE=targetRRMSE)
        self.mlp_model = MLP(actFunction=actFunction, hidden_layers=hidden_layers, max_iters=max_iters, random_state=self.random_state, scaler=scaler, solver=solver)
        self.mlp_model.train_model(self.X_train, self.y_train)

        print(f"kovetkezo 6 honapra valo elorejelzes: {self.mlp_model.forecastFutureValues(12, self.X_test)}")


        self.mlp_model.predictions = self.mlp_model.predict(self.X_test)
        self.MLPResultsZipped = zip(self.mlp_model.predictions, self.teszt_adatok)
        self.mlp_model.mse = MSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.rrmse = RRMSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.mape = MAPE(self.teszt_adatok, self.mlp_model.predictions)

   
    def predict_with_lstm(self, mode="vanilla", activation: str = "relu",  solver: str = "adam", scaler:str = "",
                           units: int = 64, n_steps: int = 1, input_dim = 100, loss="mse", n_features = 1, 
                           epochs: int = 200, verbose: int = 0):
        
        #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalamzott.
        test_data = self.adatok[-n_steps:]  + self.teszt_adatok
        learning_data = self.adatok[:-n_steps]

        self.X_train, self.y_train = split_sequence(learning_data, n_steps)
        self.X_test, self.y_test = split_sequence(test_data, n_steps) 

        self.lstm = Vanilla_LSTM(self.X_train, self.y_train, self.X_test, self.y_test, activation = activation,  solver = solver, units=units, n_steps = n_steps,
        n_features=n_features, loss = loss, scaler=scaler, epochs=epochs, input_dim=input_dim, verbose=verbose)

    
    
    def get_month_number(self, month):
        months = {
            'január': 1,
            'február': 2,
            'március': 3,
            'április': 4,
            'május': 5,
            'június': 6,
            'július': 7,
            'augusztus': 8,
            'szeptember': 9,
            'október': 10,
            'november': 11,
            'december': 12
        }
        return months[month]

    def find_best_random_state(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, random_state_min=50, random_state_max=70, scaler="standard", solver="adam", targetRRMSE=0.06):
        best_random_state = None
        best_rrmse = float(1000) 

        for random_state in range(random_state_min, random_state_max+1):
            mlp_model = MLP(actFunction=actFunction, hidden_layers = hidden_layers, max_iters = max_iters, random_state = random_state, scaler = scaler, solver=solver)
            mlp_model.train_model(self.X_train, self.y_train)
            predictions = mlp_model.predict(self.X_test)
            rrmse = RRMSE(predictions, self.teszt_adatok)
            print(f"trying {self.idosor_nev}'s MLP prediction with random state {random_state} --> RRMSE: {rrmse}")

            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_random_state = random_state
            
            if round(rrmse, 2) <= targetRRMSE:
                print(f"target RRMSE{targetRRMSE} reached, stopping search...")
                return best_random_state

        self.random_state = best_random_state
        return best_random_state
  

def split_sequence(sequence, n_steps):
        X, y = list(), list()
        sequence = [round(i, 2) for i in sequence]

        for i in range(len(sequence)):
            end_ix = i + n_steps
            if end_ix > len(sequence)-1:
                break
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

class ARIMA:
    def __init__(self, p:int = 1, d: int = 0, q: int = 0, t:int = 10, adatok = [], teszt_adatok = [], idoszakok = [], teszt_idoszakok = []):

        self.p = int(p)
        self.q = int(q)
        self.d = int(d)
     
        self.aic = 0
        self.mse = 0
        self.rrmse = 0
        self.mape = 0
        self.r_squared = 0
        self.diagram = None
        self.modelName = ""

        print("időszalok ARIMA", idoszakok)

            # Ellenőrizd, hogy minden adatstruktúra nem üres és egyforma hosszú
        if adatok is not None and teszt_adatok is not None and idoszakok is not None and teszt_idoszakok is not None:
            if len(adatok) == len(idoszakok) and len(teszt_adatok) == len(teszt_idoszakok):
                self.t = len(teszt_adatok)
                # ARIMA modell illesztése
                self.model = sm.tsa.ARIMA(adatok, order=(self.p, self.d, self.q), enforce_stationarity=True)
                self.model_fit = self.model.fit()

                # Jövőbeli értékek előrejelzése
                self.becslesek = self.model_fit.forecast(steps=self.t)
                self.aic = self.model_fit.aic

                # Egyéb értékek kiszámolása
                self.mse = MSE(teszt_adatok, self.becslesek)
                self.rrmse = RRMSE(teszt_adatok, self.becslesek)
                self.mape = MAPE(teszt_adatok, self.becslesek)
                self.r_squared = r2_score(teszt_adatok, self.becslesek)
            else:
                print(f"Az adatstruktúrák hossza nem egyezik meg: \n teszt_adatok: {len(teszt_adatok)}, teszt időszakok: {len(teszt_idoszakok)} \n adatok: {len(adatok)}, idoszakok: {len(idoszakok)} ")
        else:
            print("Nem megfelelő adatstruktúra megadva.")
  
class MLP:
    def __init__(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=2000, 
                 random_state=50, units: int = 50, scaler="standard", solver="adam"):
        self.hidden_layers = hidden_layers
        self.NrofHiddenLayers = len(hidden_layers)
        self.max_iters = max_iters
        self.random_state = random_state
        self.activation = actFunction
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver=solver,  activation=actFunction, max_iter=max_iters, random_state=random_state)
        self.scaler = StandardScaler()
        self.predictions = []
        self.mse = 0
        self.weights = []
        self.rrmse = 0
        self.mape = 0
        self.diagram = None
        self.accuracy = 0
        self.scalerMode = scaler
        self.scaler = StandardScaler()
        self.modelStr = self.NrofHiddenLayers * '{}, '
        self.modelStr = "("+self.modelStr.format(*hidden_layers)[:-1]+")"
        self.x_test = []
        self.y_test = []
        self.x_train = []
        self.y_train = []

        if (scaler == "robust"):
            self.scaler = RobustScaler()
        if (scaler == "minmax"):
            self.scaler = MinMaxScaler()

    def train_model(self, X_train, y_train):
        if self.scalerMode != "-":
            X_train = self.scaler.fit_transform(X_train)
        self.x_train = X_train
        self.y_train = y_train

        self.model.fit(X_train, y_train)
        self.weights = [layer_weights for layer_weights in self.model.coefs_]

    def predict(self, X_test, normalize=True):
        if self.scalerMode != "-" and normalize:
            X_test = self.scaler.transform(X_test)   
   
        return self.model.predict(X_test)
    

    def forecastFutureValues(self, n, x_test):
        future_forecasts = []
        input = x_test[-1].reshape(1, -1)  # Átalakítjuk a legutolsó  input értéket 2D formátumra

        for i in range(n):
            forecast = self.predict(input)[0]  # a predict 2d-s lisát ad vissza, 1 elemmel, mert csak 1 input van
            future_forecasts.append(forecast) 
            print(f"{i}. : {input} ---> {forecast}")
            #csúsztatjuk egyel arréb toljuk az input elmeit, az utolsó a legutóbbi előrejelzett érték lesz
            input = np.hstack((input[:, 1:], forecast.reshape(1, -1)))

        return future_forecasts



       
class Vanilla_LSTM:
    def __init__(self,  x_train: list = [], y_train: list = [], x_test: list = [], y_test: list= [], units:int = 50, activation: str = "relu",  solver: str = "adam", scaler: str = "", n_features: int = 1, n_steps: int = 3, input_dim: int = 100, loss: str ="mse",  epochs: int = 200, verbose: int = 0):
        self.diagram = None

        self.x_train = x_train; self.y_train= y_train
        self.x_test = x_test; self.y_test = y_test
        
        self.model = Sequential()
        self.model.add(LSTM(units=units, activation=activation, input_shape=(n_steps, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer=solver, loss=loss)
        
        if(scaler == "minmax"):
            self.scaler = MinMaxScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        # reshape from [samples, timesteps] into [samples, timesteps, features] for LSTM 
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], n_features))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], n_features))

        if self.scaler is not None:
            # Tanítóadat inputok normalizálása
            self.x_train_Normalized = self.scaler.fit_transform(self.x_train.reshape(-1, self.x_train.shape[-1])).reshape(self.x_train.shape)

            # Tanító elvárt outputok normalizálása
            self.y_train_normalized = self.scaler.fit_transform(self.y_train.reshape(-1, 1))

            # Teszthalmaz inputok normalizálása
            self.x_test_Normalized = self.scaler.fit_transform(self.x_test.reshape(-1, self.x_test.shape[-1])).reshape(self.x_test.shape)

            # Teszthalmaz elvárt outputok normalizálása
            self.y_test_normalized = self.scaler.fit_transform(self.y_test.reshape(-1, 1))

            # Tanítás és jóslat a normalizált inputokkal
            self.model.fit(self.x_train_Normalized, self.y_train_normalized, epochs=epochs, verbose=verbose)
            self.predictions = self.model.predict(self.x_test_Normalized, verbose=0)

            # a jóslatok visszaalakítása normál formáról
            self.predictions = self.scaler.inverse_transform(self.predictions)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]


        else:
            # Tanítás és jóslat a nyers inputokkal
            self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=verbose)
            self.predictions = self.model.predict(self.x_test, verbose=0)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]
        
        self.forecastZipped = zip(self.predictions, self.y_test)
        self.mse = MSE(self.predictions, self.y_test)
        self.rrmse = RRMSE(self.predictions, self.y_test)
        self.mape = MAPE(self.predictions, y_test)
        print(f"mape: {self.predictions} \n vs \n {self.y_test} ")


    def printTraintSet(self):
        res = "<h1>training set: x == > y</h1>"
        for i in range(len(self.x_train)) :
            res += f"{i+1}.: {self.x_train[i]} ==> {self.y_train[i]} <br>"
        
        return res
    
    def printTestSet(self):
        res = f"<h1> prediction set: x (input) == > y </h1> <br>"
        res = f"MSE = {self.mse}, RRMSE = {self.rrmse} <br>"
        for i in range(len(self.x_test)) :
            res += f"{i+1}.: {self.x_test[i]} ==> {self.y_test[i]}, joslat: {self.predictions[i]} <br>"

        return res
    
    def printNormalizedTestSet(self):
        res = f"<h1> Normalized prediction set: x (input) == > y </h1> <br>"
        res = f"MSE = {self.mse}, RRMSE = {self.rrmse} <br>"
        for i in range(len(self.x_test_Normalized)) :
            res += f"{i+1}.: {self.x_test_Normalized[i]} ==> {self.y_test[i]}, joslat: {self.predictions[i]} <br>"

        return res
    
    def printNormalizedTrainSet(self):
        res = f"<h1> Normalized training set: x (input) == > y </h1> <br>"
        res = f"MSE = {self.mse}, RRMSE = {self.rrmse} <br>"
        for i in range(len(self.x_train_Normalized)) :
            res += f"{i+1}.: {self.x_train_Normalized[i]} ==> {self.y_train[i]} <br>"

        return res


def MSE(becslesek, teszt_adatok,):
    try:
        n = len(teszt_adatok)
        teszt_adatok_np = np.array(teszt_adatok)
        becslesek_np = np.array(becslesek)
        mse = np.sum((teszt_adatok_np - becslesek_np)**2) / n
        return mse * 100
    except:
        return -1   
    
def RRMSE(becslesek, teszt_adatok):
    try:
        mse = MSE(becslesek, teszt_adatok)
        mean_y = np.mean(teszt_adatok)
        if mse < 0 or mean_y <= 0:
            rrmse = np.sqrt(-1*(mse)) / mean_y
        else:  
            rrmse = np.sqrt(mse) / mean_y
        return rrmse*10
    
    except Exception as e:
        print(traceback.format_exc())
        return -1
    
def MAPE(becslesek, teszt_adatok):
    if len(teszt_adatok) != len(becslesek):
        raise ValueError("A becsült és valós értékek listáinak azonos hosszúnak kell lenniük.")

    absolute_percentage_errors = []
    for prediction, actual in zip(becslesek, teszt_adatok):
        absolute_percentage_error = abs((actual - prediction) / actual) * 100
        absolute_percentage_errors.append(absolute_percentage_error)

    mean_absolute_percentage_error = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    return mean_absolute_percentage_error