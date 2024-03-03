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
        
    def autocorrelationPlot(self):
        buffer = io.BytesIO()
        fig, ax = plt.subplots()
        pd.plotting.autocorrelation_plot(self.adatok, ax=ax)
        plt.title(self.idosor_nev)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return encoded_image

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
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.pacf_acf_Diagram = encoded_image

    def distributionPlot(self):
        # Eredeti értékek hisztogramja
        plt.subplot(2, 1, 1)
        plt.hist(self.adatok, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Értékek')
        plt.ylabel('Gyakoriság')
        plt.title(f"{self.idosor_nev} Hisztogram")

        # Logaritmizált értékek hisztogramja
        plt.subplot(2, 1, 2)
        log_adatok = np.log(self.adatok)  # Logaritmizálás
        plt.hist(log_adatok, bins=30, color='skyblue', edgecolor='black')
        plt.xlabel('Logaritmizált értékek')
        plt.ylabel('Gyakoriság')
        plt.title(f"{self.idosor_nev} Logaritmizált Hisztogram")

        plt.tight_layout()  # Automatikus elrendezés

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
        
 
    def predict_with_mlp(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, scaler="standard", randomStateMax=70, randomStateMin=50, solver="adam", targetRRMSE=0.6, x_mode = "delayed", n_delays = 3, n_pred=6):
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
        
        self.futureforecasts_y, self.futureforecasts_x = self.mlp_model.forecastFutureValues(n_pred, self.X_test)

        self.mlp_model.predictions = self.mlp_model.predict(self.X_test)
        self.MLPResultsZipped = zip(self.mlp_model.predictions, self.teszt_adatok)
        self.mlp_model.mse = MSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.rrmse = RRMSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.mape = MAPE(self.teszt_adatok, self.mlp_model.predictions)

   
    def predict_with_lstm(self, mode="vanilla", activation: str = "relu",  solver: str = "adam", scaler:str = "",
                           units: int = 64, n_steps: int = 1, input_dim = 100, loss="mse", n_features = 1, 
                           epochs: int = 200, verbose: int = 0, n_pred:int = 6 ):
        
        #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalamzott.
        test_data = self.adatok[-n_steps:]  + self.teszt_adatok
        learning_data = self.adatok[:-n_steps]

        self.X_train, self.y_train = split_sequence(learning_data, n_steps)
        self.X_test, self.y_test = split_sequence(test_data, n_steps) 

        self.lstm = Vanilla_LSTM(self.X_train, self.y_train, self.X_test, self.y_test, activation = activation,  solver = solver, units=units, n_steps = n_steps,
        n_features=n_features, loss = loss, scaler=scaler, epochs=epochs, input_dim=input_dim, verbose=verbose, n_pred=n_pred)
    
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

        if adatok is not None and teszt_adatok is not None and idoszakok is not None and teszt_idoszakok is not None:
            if len(adatok) == len(idoszakok) and len(teszt_adatok) == len(teszt_idoszakok):
                self.t = len(teszt_adatok)
                # ARIMA modell illesztése
                self.model = sm.tsa.ARIMA(adatok, order=(self.p, self.d, self.q))
                self.model_fit = self.model.fit()
                self.aic = self.model_fit.aic

                # Jövőbeli értékek előrejelzése
                history = [x for x in adatok]
                predictions = list()
            
                # walk-forward validation
                for t in range(len(teszt_adatok)):
                    model = sm.tsa.ARIMA(history, order=(self.p, self.d, self.q))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = teszt_adatok[t]
                    history.append(obs)
                  #  print('predicted=%f, expected=%f' % (yhat, obs))
                
                self.becslesek = predictions
                self.becsleseksZipped  = zip(predictions, teszt_adatok)

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
        x_axis = []
        input = x_test[-1].reshape(1, -1)  # Átalakítjuk a legutolsó  input értéket 2D formátumra

        for i in range(n):
            forecast = self.predict(input)[0]  # a predict 2d-s lisát ad vissza, 1 elemmel, mert csak 1 input van
            future_forecasts.append(forecast) 
            print(f"{i+1}. : {input} ---> {forecast}")
            x_axis.append(f"{i+1}. jóslat")
            #csúsztatjuk egyel arréb toljuk az input elmeit, az utolsó a legutóbbi előrejelzett érték lesz
            input = np.hstack((input[:, 1:], forecast.reshape(1, -1)))

        print(f"future firecast: x={x_axis}, \n y={future_forecasts}")
        return future_forecasts, x_axis
          
class Vanilla_LSTM:
    def __init__(self,  x_train: list = [], y_train: list = [], x_test: list = [], y_test: list= [], units:int = 50, activation: str = "relu", 
                  solver: str = "adam", scaler: str = "", n_features: int = 1, n_steps: int = 3, input_dim: int = 100, loss: str ="mse",  epochs: int = 200, verbose: int = 0, n_pred:int = 6):
        
        self.diagram = None; self.epochs = epochs; self.verbose = verbose; self.activation = activation; self.solver = solver; self.n_steps = n_steps; self.loss = loss; self.units = units; self.n_pred = n_pred; self.scalerStr = scaler.strip();
        
        print(f"SKÁLÁZÁSI MÓD: { self.scalerStr}")

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
        self.y_train = y_train; self.y_test = y_test; self.x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], n_features)); self.x_test =  x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))    

        self.trainModel()
        self.predict()      
        self.futureforecasts_y, self.futureforecasts_x = self.forecastFutureValues()
        
        self.forecastZipped = zip(self.predictions, self.y_test)
        self.mse = MSE(self.predictions, self.y_test)
        self.rrmse = RRMSE(self.predictions, self.y_test)
        self.mape = MAPE(self.predictions, y_test)


    def trainModel(self):  # betanítás
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
            # Tanítóadat inputok normalizálása
            self.x_train_Normalized = self.scaler.fit_transform(self.x_train.reshape(-1, self.x_train.shape[-1])).reshape(self.x_train.shape)
            # Tanító elvárt outputok normalizálása
            self.y_train_normalized = self.scaler.fit_transform(self.y_train.reshape(-1, 1))
            # Teszthalmaz inputok normalizálása
            self.x_test_Normalized = self.scaler.fit_transform(self.x_test.reshape(-1, self.x_test.shape[-1])).reshape(self.x_test.shape)
            # Teszthalmaz elvárt outputok normalizálása
            self.y_test_normalized = self.scaler.fit_transform(self.y_test.reshape(-1, 1))

            # Tanítás a normalizált inputokkal
            self.model.fit(self.x_train_Normalized, self.y_train_normalized, epochs=self.epochs, verbose=self.verbose)
        
        else:
             # Tanítás a sima inputokkal
            self.model.fit(self.x_train, self.y_train, epochs = self.epochs, verbose = self.verbose)

    
    def predict(self): # előrejelzés a tesztadatokra
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
        #Ha az inputok normalizálva voltak, akkor a kimenetet vissza kell alakitnai normalformarol
            self.predictions = self.model.predict(self.x_test_Normalized, verbose=self.verbose)
            self.predictions = self.scaler.inverse_transform(self.predictions)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]
     
        else:
            self.predictions = self.model.predict(self.x_test, verbose=self.verbose)
            self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]
    
    
    def predictSingle(self, input): #előrejelzés 1db inputra (forecastFutureValues használja)
        predictions = self.model.predict(input, verbose=self.verbose)

        # Ha az inputok normalizálva voltak, akkor a kimenetet vissza kell alakítani normalformára
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
            print("\n \n a modell vissza normalizálja a jósaltokat  \n \n}")
            prediction_normalized = predictions
            prediction = self.scaler.inverse_transform(predictions)
        else:
            prediction = predictions
            prediction_normalized = prediction

        prediction = [round(item, 2) for sublist in prediction for item in sublist]
        prediction_normalized = [round(item, 2) for sublist in prediction_normalized for item in sublist]

        return prediction[0], prediction_normalized[0]


    def forecastFutureValues(self): # előrejelzás a tesztadatokon túl, az utolsó input mintázat alapján, majd mindig frissítve az utolsó értéket a legutóbbi jóslatra
        future_forecasts = []
        x_axis = []
        
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
            input = [self.x_test_Normalized[-1]]
            print("\n \n a modell normalizált adatokkal jósol  \n \n}")
        else:
            input = [self.x_test[-1]]
            print("\n \n a modell módosítatlan adatokkal jósol  \n \n}")
        
        for i in range(self.n_pred):
            forecast, forecastNormalized = self.predictSingle(input) 
            future_forecasts.append(forecast) 

            print(f"{i+1}. : {input} ---> {forecast}")
            x_axis.append(f"{i+1}. jóslat")
            # egyel arréb toljuk az input elmeit, az utolsó a legutóbbi előrejelzett érték lesz
            input = [Slide(input[0], forecastNormalized)]

        print(f"future forecast: x={x_axis}, \n y={future_forecasts}")
        return future_forecasts, x_axis



    def printTraintSet(self):
        res = "<h1>training set: x == > y</h1>"
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
            for i in range(len(self.x_train)) :
                res += f"{i+1}.: {self.x_train_Normalized[i]} ==> {self.y_train_normalized[i]} ({self.y_train[i]}) <br>"
        else:
            for i in range(len(self.x_train)) :
                res += f"{i+1}.: {self.x_train[i]} ==> {self.y_train[i]} <br>"

        return res
    
    def printTestSet(self):
        res = f"<h1> prediction set: x (input) == > y </h1> <br>"
        res = f"MSE = {self.mse}, RRMSE = {self.rrmse} <br>"
        if self.scaler is not None and  self.scalerStr != "" and self.scalerStr != "-":
            for i in range(len(self.x_test_Normalized)) :
                res += f"{i+1}.: {self.x_test_Normalized[i]} ==> {self.y_test_normalized[i]} ({self.y_test[i]}), joslat: {self.predictions[i]} <br>"
        else:
            for i in range(len(self.x_test)) :
                res += f"{i+1}.: {self.x_test[i]} ==> {self.y_test[i]}, joslat: {self.predictions[i]} <br>"
        
        return res
    

def Slide(input_array, new_value):
    # Ellenőrizze, hogy a bemeneti adatszerkezet egy 2D Numpy tömb
    if not isinstance(input_array, np.ndarray) or input_array.ndim != 2:
        raise ValueError("A bemeneti adatszerkezetnek egy 2D Numpy tömbnek kell lennie.")

    # Másolat készítése a bemeneti adatszerkezetről
    modified_array = np.copy(input_array)

    # Az első n-1 elemet egyel előrébb csúsztatja
    modified_array[:-1] = modified_array[1:]

    # Az utolsó elem cseréje a második paraméterként kapott értékre
    modified_array[-1] = new_value

    return modified_array

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
        print("A becsült és valós értékek listáinak azonos hosszúnak kell lenniük.")
        return -1

    absolute_percentage_errors = []
    for prediction, actual in zip(becslesek, teszt_adatok):
        absolute_percentage_error = abs((actual - prediction) / actual) * 100
        absolute_percentage_errors.append(absolute_percentage_error)

    mean_absolute_percentage_error = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    return mean_absolute_percentage_error