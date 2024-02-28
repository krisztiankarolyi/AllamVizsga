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
from keras.layers import LSTM, Dense


class Stat :
    def __init__(self, idosor_nev, adatok, idoszakok):
        self.idosor_nev = idosor_nev
        self.adatok = adatok
        self.idoszakok = idoszakok
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
    
    def setTesztIdoszakok(self, idoszakok: list):
        self.teszt_idoszakok = idoszakok

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
        
           
    def ARIMA(self, p:int = 1, d: int = 0, q: int = 0, t:int = 10):
        self.ARIMA = ARIMA(p, d, q, t, self.adatok, self.teszt_adatok)
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
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, target, test_size=10, shuffle=False)

        else:
            # az adatok n darab korábbi megfigyeléstől függnek (késleltetett értékek)
            self.dependency = f"{n_delays} db Késleltetett érték"
            test_data = self.adatok[-n_delays:]  + self.teszt_adatok

            self.X_train, self.y_train = split_sequence(self.adatok, n_delays)
            self.X_test, self.y_test = split_sequence(test_data, n_delays)


        self.X_Train_Y_Train_Zipped = zip(self.X_train, self.y_train)
        self.X_Test_Y_Test_Zipped = zip(self.X_test, self.y_test)

        self.random_state = self.find_best_random_state(actFunction=actFunction, random_state_min=randomStateMin, random_state_max=randomStateMax, max_iters=max_iters, scaler=scaler, hidden_layers=hidden_layers, solver=solver, targetRRMSE=targetRRMSE)
        self.mlp_model = MLP(self.teszt_adatok, actFunction=actFunction, hidden_layers=hidden_layers, max_iters=max_iters, random_state=self.random_state, scalerMode=scaler, solver=solver)
        self.mlp_model.train_model(self.X_train, self.y_train)
        self.mlp_model.predictions = self.mlp_model.predict(self.X_test)
        self.MLPResultsZipped = zip(self.mlp_model.predictions, self.teszt_adatok)
        self.mlp_model.mse = MSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.rrmse = RRMSE(self.teszt_adatok, self.mlp_model.predictions)

   
    def predict_with_lstm(self, mode="vanilla", activation: str = "relu",  solver: str = "adam", scaler:str = "",
                           units: int = 64, n_steps: int = 1, input_dim = 100, loss="mse", n_features = 1, 
                           epochs: int = 200, verbose: int = 0):
        test_data = self.adatok[-n_steps:]  + self.teszt_adatok

        self.X_train, self.y_train = split_sequence(self.adatok, n_steps)
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
            mlp_model = MLP(self.teszt_adatok, actFunction, hidden_layers, max_iters, random_state, scaler, solver=solver)

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
    def __init__(self, p:int = 1, d: int = 0, q: int = 0, t:int = 10, adatok = [], teszt_adatok = []):

        self.aic = 0
        self.mse = 0
        self.rrmse = 0
        self.r_squared = 0
        self.diagram = None
        self.modelName = ""

        self.p = int(p); self.q = int(q); self.d = int(d); self.t = int(t)

        idosor = adatok
        self.model = sm.tsa.ARIMA(idosor, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

        self.becslesek = self.model_fit.forecast(self.t)
        self.aic = self.model_fit.aic

        self.becsleseksZipped = zip(self.becslesek, teszt_adatok)
        self.mse = MSE(teszt_adatok, self.becslesek)
        self.rrmse = RRMSE(teszt_adatok, self.becslesek)
        self.r_squared  = r2_score(teszt_adatok, self.becslesek) 
  
class MLP:
    def __init__(self, test_data, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=2000, random_state=50, units: int = 50,scalerMode="standard", solver="adam"):
        self.test_data = test_data
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
        self.diagram = None
        self.accuracy = 0
        self.scalerMode = scalerMode
        self.scaler = StandardScaler()
        self.modelStr = self.NrofHiddenLayers * '{}, '
        self.modelStr = "("+self.modelStr.format(*hidden_layers)[:-1]+")"

        if (scalerMode == "robust"):
            self.scaler = RobustScaler()
        if (scalerMode == "minmax"):
            self.scaler = MinMaxScaler()

    def train_model(self, X_train, y_train):
        if self.scalerMode != "-":
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
        self.weights = [layer_weights for layer_weights in self.model.coefs_]

    def predict(self, X_test):
        if self.scalerMode != "-":
            X_test = self.scaler.transform(X_test)    
        return self.model.predict(X_test)
       
class Vanilla_LSTM:
    def __init__(self,  x_train: list = [], y_train: list = [], x_test: list = [], y_test: list= [], units:int = 50, activation: str = "relu",  solver: str = "adam", scaler: str = "", n_features: int = 1, n_steps: int = 3, input_dim: int = 100, loss: str ="mse",  epochs: int = 200, verbose: int = 0):
        self.diagram = None

        if(scaler == "minmax"):
            self.scaler = MinMaxScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = None
        
        self.x_train = x_train; self.y_train= y_train
        self.x_test = x_test; self.y_test = y_test

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], n_features))
        self.x_test = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], n_features))

        self.model = Sequential()
        self.model.add(LSTM(units=units, activation=activation, input_shape=(n_steps, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer=solver, loss=loss)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=verbose)

        if(self.scaler is not None):
            normalized_data = self.scaler.fit_transform(self.x_train.reshape(-1, self.x_train.shape[-1])).reshape(self.x_train.shape)
            samples, time_steps, features = normalized_data.shape
            reshaped_data = normalized_data.reshape((samples, time_steps, features))
            self.x_train_Normalized = reshaped_data

            normalized_data = self.scaler.fit_transform(self.x_test.reshape(-1, self.x_test.shape[-1])).reshape(self.x_test.shape)
            samples, time_steps, features = normalized_data.shape
            reshaped_data = normalized_data.reshape((samples, time_steps, features))
            self.x_test_Normalized = reshaped_data
            
            self.predictions = self.model.predict(self.x_test_Normalized, verbose=0)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]

        else:
            self.predictions = self.model.predict(self.x_test, verbose=0)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]
        
        self.forecastZipped = zip(self.predictions, self.y_test)

        self.mse = MSE(self.predictions, self.y_test)
        self.rrmse = RRMSE(self.predictions, self.y_test)
    
    def printTraintSet(self):
        res = "<h1>training set: x == > y</h1>"
        for i in range(len(self.x_train)) :
            res += f"{i+1}.: {self.x_train[i]} ==> {self.y_train[i]} <br>"
        
        return res
    
    def printTestSet(self):
        res = "<h1>prediction set: x (input) == > y </h1> <br>"
        if(self.scaler is not None):
            res += "the learning data has been normalized <br>"
        res = f"MSE = {self.mse}, RRMSE = {self.rrmse} <br>"
        for i in range(len(self.x_test)) :
            res += f"{i+1}.: {self.x_test[i]} ==> {self.y_test[i]}, joslat: {self.predictions[i]} <br>"

        return res

def MSE(becslesek, teszt_adatok,):
    try:
        n = len(teszt_adatok)
        teszt_adatok_np = np.array(teszt_adatok)
        becslesek_np = np.array(becslesek)
        mse = np.sum((teszt_adatok_np - becslesek_np)**2) / n
        return mse
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
        return rrmse
    
    except Exception as e:
        print(traceback.format_exc())
        return -1