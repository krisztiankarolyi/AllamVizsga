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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from scipy.stats import kstest
import pandas as pd
from keras.layers import LSTM, Dense
from sklearn.metrics import confusion_matrix
from statsmodels.stats.diagnostic import het_white
import seaborn as sns

class Stat :
    def __init__(self, idosor_nev, adatok, idoszakok):
        self.idosor_nev = idosor_nev
        self.adatok = adatok
        self.idoszakok = idoszakok
        self.adf = {}; self.kpss = {}
        self.teszt_idoszakok = []
        self.Kolmogorov_Smirnov = {'statisztika': 0, 'p_value': 0}
        self.log_Kolmogorov_Smirnov = {'statisztika': 0, 'p_value': 0}

    def calculateStatistics(self):
        self.atlag = round(np.mean(self.adatok), 2)
        self.szoras = round(np.std(self.adatok), 2)
        self.variancia = round(np.var(self.adatok), 2)
        self.median = round(np.median(self.adatok), 2)
        self.min = np.min(self.adatok)
        self.max = np.max(self.adatok)
        self.minDatum = self.idoszakok[list.index(self.adatok, self.min)]
        self.maxDatum = self.idoszakok[list.index(self.adatok, self.max)]
        self.StationarityTest()
        self.distributionPlot = self.distributionPlot()

        ks_statistic, p_value = kstest(self.adatok, 'norm')
        self.Kolmogorov_Smirnov['statisztika'] = ks_statistic
        self.Kolmogorov_Smirnov['p_value'] = p_value

        ks_statistic, p_value = kstest(np.log(self.adatok), 'norm')
        self.log_Kolmogorov_Smirnov['statisztika'] = ks_statistic
        self.log_Kolmogorov_Smirnov['p_value'] = p_value

    def setTesztAdatok(self, teszt_adatok: list):
        self.teszt_adatok = teszt_adatok
    
    def setTesztIdoszakok(self, idoszakok: list):
        self.teszt_idoszakok = idoszakok

    def StationarityTest(self):
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

    def predictARIMA(self, p:int = 1, d: int = 0, q: int = 0, n_pred:int = 6):
        t = len(self.teszt_adatok)
        self.ARIMA = ARIMA(p, d, q, adatok=self.adatok, teszt_adatok=self.teszt_adatok, idoszakok=self.idoszakok, teszt_idoszakok=self.teszt_idoszakok, n_pred = n_pred )
        self.ARIMA.fit(self.adatok)
        self.ARIMA.predict(self.teszt_adatok, self.adatok)
        try:
            self.ARIMA.residuals = np.array([self.teszt_adatok[i] - self.ARIMA.becslesek[i] for i in range(len(self.teszt_adatok))])
            self.ARIMA.errorHistogram = plot_error_analysis(residuals=self.ARIMA.residuals, name=self.idosor_nev+" "+self.ARIMA.modelName)
            self.ARIMA.residualsPlot = plot_Residuals(residuals=self.ARIMA.residuals, name=self.idosor_nev+" "+self.ARIMA.modelName)
            self.ARIMA.white = White(self.ARIMA.residuals)
            self.ARIMA.resACFPlot = self.acfPlot(self.ARIMA.residuals)
            self.ARIMA.becsleseksZipped  = zip(self.ARIMA.becslesek, self.teszt_adatok, self.ARIMA.residuals)
            
        except Exception as exp:
            print("Hiba történt")
            print(traceback.format_exc())

        return self.ARIMA
        
    def plot_acf_and_pacf(self, x: list = [], pacf: bool = True):
        if len(x) == 0:
            x = self.adatok
        fig, (ax1, ax2) = plt.subplots(2 if pacf else 1, 1, figsize=(6, 6), sharex=False)
        fig.subplots_adjust(hspace=0.3)
        
        max_lags = len(x) // 2 if len(x) > 12 else len(x)
        lags = min(max_lags, 20)  # Set maximum lags to 20 or half the sample size, whichever is smaller
        
        plot_acf(x, lags=lags, ax=ax1, title=f"Autokorreláció ({self.idosor_nev})")
        
        if pacf:
            plot_pacf(x, lags=lags, ax=ax2, title=f"Parciális Autokorreláció ({self.idosor_nev})")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        self.pacf_acf_Diagram = encoded_image
        return encoded_image
    
    def acfPlot(self, x: list = []):
        x = np.array(x)
        print(f"resiudals to be plotted:  {x}")
        lags = len(x) // 2 if len(x) > 12 else len(x)-1
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_acf(x, lags=lags, ax=ax)
        ax.set_title("Reziduumok autokorrelációi")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    
    def distributionPlot(self):
        bins = int(max(self.adatok) - min(self.adatok))
        n, bins, _ = plt.hist(self.adatok, bins=bins, color='blue', edgecolor='black')
        plt.xlabel('Értékek')
        plt.ylabel('Gyakoriság')
        plt.title(f"{self.idosor_nev} Hisztogram")
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        # Az x tengely feliratainak beállítása a létrehozott csoportokra
        plt.xticks(bins)

        plt.plot(bins[:-1], n, color='red', marker='o', linestyle='-', linewidth=2, markersize=6)

        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
        
    def predict_with_mlp(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, 
                         scalerMode="standard", randomStateMax=70, randomStateMin=50, solver="adam", 
                         targetMSE=0.1, x_mode = "delayed", n_delays = 3, n_pred=6, normOut: bool = False):
        if not self.teszt_adatok:
            print("Nincsenek tesztelési adatok.")
            return          
        
        # az adatok n darab korábbi megfigyeléstől függnek (késleltetett értékek)
        #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalamzott.
        test_data = self.adatok[-n_delays:]  + self.teszt_adatok
        learning_data = self.adatok[:-n_delays]
    
        self.X_train, self.y_train = split_sequence(learning_data, n_delays)
        self.X_test, self.y_test = split_sequence(test_data, n_delays)
        
        # normalizálás ha kell 
        scaler = None
        if (scalerMode == "robust"):
            scaler = RobustScaler()
            scaler = RobustScaler()
        if (scalerMode == "minmax"):
           scaler = MinMaxScaler()
           scaler = MinMaxScaler()
        if(scalerMode == "standard"):
            scaler = StandardScaler()
            scaler = StandardScaler()

        if scalerMode != "-":
            if(scalerMode == "log"):
                self.X_train = np.log(self.X_train)
                self.X_test = np.log(self.X_test)
                if(normOut):
                    self.y_test = np.log(self.y_test)
                    self.y_train = np.log(self.y_train)
            else:
                self.X_train = scaler.fit_transform(np.array(self.X_train))
                self.X_test = scaler.transform(np.array(self.X_test))
                if(normOut):
                    self.y_test = scaler.fit_transform(self.y_test.reshape(-1, 1))
                    self.y_train = scaler.transform(self.y_train.reshape(-1, 1))

                    self.y_test = [item for sublist in self.y_test for item in sublist]
                    self.y_train =[item for sublist in self.y_train for item in sublist]

                
        print(f"\n --------------------------------- \n x_train = {self.X_train}")
        print(f"\n --------------------------------- \n y_train = {self.y_train}")
        print(f"\n --------------------------------- \n x_test = {self.X_test}")
        print(f"\n --------------------------------- \n y_test = {self.y_test}")

        self.X_Train_Y_Train_Zipped = zip(self.X_train, self.y_train)
        self.X_Test_Y_Test_Zipped = zip(self.X_test, self.y_test)

        self.random_state = self.find_best_random_state(y_train=self.y_train, x_train=self.X_train, actFunction=actFunction, random_state_min=randomStateMin, 
                                                        random_state_max=randomStateMax, max_iters=max_iters,
                                                         scaler=scaler, hidden_layers=hidden_layers, solver=solver, targetMSE=targetMSE, y_test=self.y_test)
        
        self.mlp_model = MLP(actFunction=actFunction, hidden_layers=hidden_layers, max_iters=max_iters, random_state=self.random_state, scaler=scaler, scalerMode=scalerMode, solver=solver)
        self.mlp_model.train_model(self.X_train, self.y_train)
        self.futureforecasts_y, self.futureforecasts_x = self.mlp_model.forecastFutureValues(n_pred, self.X_test, scaler=scaler, scalerMode=scalerMode, normOut=normOut)

        self.mlp_model.predictions = self.mlp_model.predict(self.X_test)
        # visszatranstformálás ha kell
        if scalerMode != "-" and normOut:
            if(scalerMode == "log"):
                self.mlp_model.predictions = np.exp(self.mlp_model.predictions)
            else:
                self.mlp_model.predictions = scaler.inverse_transform(self.mlp_model.predictions.reshape(-1, 1))
                self.mlp_model.predictions = [item for sublist in self.mlp_model.predictions for item in sublist]
       
        self.mlp_model.predictions = np.array(self.mlp_model.predictions)

        self.mlp_model.mse = MSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.rrmse = RRMSE(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.r2 = r2_score(self.teszt_adatok, self.mlp_model.predictions)
        self.mlp_model.mape = MAPE(self.teszt_adatok, self.mlp_model.predictions)

        self.mlp_model.residuals =  np.array([self.teszt_adatok[i] - self.mlp_model.predictions[i] for i in range(len(self.teszt_adatok))])
        self.mlp_model.white = White(self.mlp_model.residuals)
        self.mlp_model.residualsPlot = plot_Residuals(residuals=self.mlp_model.residuals, name=self.mlp_model.modelStr)
        self.mlp_model.errorHistogram = plot_error_analysis(residuals=self.mlp_model.residuals, name=self.mlp_model.modelStr)
        self.mlp_model.resACFPlot = self.acfPlot(self.mlp_model.residuals)
        self.MLPResultsZipped = zip(self.mlp_model.predictions, self.teszt_adatok, self.mlp_model.residuals)     
  
    def predict_with_lstm(self, mode="vanilla", activation: str = "relu",  solver: str = "adam", 
                          scaler:str = "", units: int = 64, n_steps: int = 1, 
                          input_dim: int = 100, loss: str ="mse", n_features: int = 1, 
                           epochs: int = 200, verbose: int = 0, n_pred:int = 6, normOut: bool = False):

        #adatok átcsoportosítása, hogy kijöjjön annyi jóslat, amennyit a test data alapból tartalmazott.
        test_data = self.adatok[-n_steps:]  + self.teszt_adatok
        learning_data = self.adatok[:-n_steps]

        self.lstm = Vanilla_LSTM(learning_data=learning_data, test_data=test_data, activation = activation,  solver = solver, units=units, n_steps = n_steps,
        n_features=n_features, loss = loss, scaler=scaler, epochs=epochs, input_dim=input_dim, verbose=verbose, n_pred=n_pred, name = self.idosor_nev, normOut = normOut)
        self.lstm.errorHistogram = plot_error_analysis(self.teszt_adatok, self.lstm.predictions)
        self.lstm.residuals = np.array([self.teszt_adatok[i] - self.lstm.predictions[i] for i in range(len(self.teszt_adatok))])
        self.lstm.residualsPlot = plot_Residuals(self.lstm.residuals)
        self.lstm.white = White(self.lstm.residuals)
        self.lstm.resACFPlot = self.acfPlot(self.lstm.residuals)
        self.lstmResultsZipped = zip(self.lstm.predictions, self.teszt_adatok, self.lstm.residuals)

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

    def find_best_random_state(self, x_train, y_train,  actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=3000, 
                               random_state_min=50, random_state_max=70, scaler="standard", solver="adam", targetMSE=0.01, y_test = []):
        best_random_state = None
        best_mse = float(1000) 
        #print(f"\n -------------------------- \n preds: {predictions} \n ------------vs------------\n target: {self.y_test} ")
       
        for random_state in range(random_state_min, random_state_max+1):
            mlp_model = MLP(actFunction=actFunction, hidden_layers = hidden_layers, max_iters = max_iters, random_state = random_state, scaler = scaler, solver=solver)
            mlp_model.train_model(x_train, y_train)
            predictions = mlp_model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            print(f"trying {self.idosor_nev}'s MLP prediction with random state {random_state} --> MSE: {mse}")

            if mse < best_mse:
                best_mse = mse
                best_random_state = random_state
            
            if round(mse, 2) <= targetMSE:
                print(f"target RRMSE{targetMSE} reached, stopping search...")
                return best_random_state

        self.random_state = best_random_state
        return best_random_state
  
    def AutoARIMA(self, data: list=[], seasonal: bool = False):
        print("------------AUTO ARIMA")
        if data == []:
            data = self.adatok

        from pmdarima import auto_arima
        import pandas as pd

        # Assuming 'data' is your time series data
        # Example: data = pd.read_csv('your_data.csv', index_col='date_column', parse_dates=True)
        # Adjust the parameters and seasonal parameters as needed
        model = auto_arima(y = data, start_p=0, start_q=0,
                        max_p=5, max_q=5, 
                        seasonal=seasonal,
                        d=1,  trace=True,
                        error_action='ignore',  
                        stationary=False,
                        suppress_warnings=True, 
                        stepwise=False)
        print(model.summary())


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
    def __init__(self, p:int = 1, d: int = 0, q: int = 0, adatok = [], teszt_adatok = [], idoszakok = [], teszt_idoszakok = [], n_pred: int = 6):
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
        self.n = n_pred
        self.errorMatrix = None
        self.errorHistogram = None
        self.teszt_idoszakok = teszt_idoszakok

        if adatok is not None and teszt_adatok is not None and idoszakok is not None and teszt_idoszakok is not None:

            if len(adatok) == len(idoszakok) and len(teszt_adatok) == len(teszt_idoszakok):
                self.t = len(teszt_adatok)   
            else:
                print(f"Az adatstruktúrák hossza nem egyezik meg: \n teszt_adatok: {len(teszt_adatok)}, teszt időszakok: {len(teszt_idoszakok)} \n adatok: {len(adatok)}, idoszakok: {len(idoszakok)} ")
        else:
            print("Nem megfelelő adatstruktúra megadva.")

    def fit(self, adatok):
        # ARIMA modell illesztése
        self.model = sm.tsa.ARIMA(adatok, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()
        self.aic = self.model_fit.aic
        self.bic = self.model_fit.bic
    
    def predict(self, teszt_adatok, adatok):
        history = [x for x in adatok]
        predictions = list()
        for t in range(len(teszt_adatok)):
            model = sm.tsa.ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            output = model_fit.forecast()
            predictions.append(output[0])
            self.standard_error = model_fit.bic
            self.conf_int = model_fit.conf_int(cols=None, alpha=0.05)
            obs = teszt_adatok[t]
            history.append(obs)
        
        self.becslesek = predictions

        # Egyéb értékek kiszámolása
        self.mse = MSE(teszt_adatok, self.becslesek)
        self.rrmse = RRMSE(teszt_adatok, self.becslesek)
        self.mape = MAPE(teszt_adatok, self.becslesek)
        try:
            self.r_squared = r2_score(teszt_adatok, self.becslesek)
        except:
            self.r_squared = "N/A"

    
    def forecastExtra(self):
        # egy lépésben előrejelzés --> kevésbé precíz az előrelépő validációval szemben, a tesztadaton túli előrejelzéshez jó
        future_predictions = self.model_fit.forecast(steps=self.n)   
        return future_predictions
  
class MLP:
    def __init__(self, actFunction="logistic", hidden_layers=(12, 12, 12), max_iters=2000, 
                 random_state=50, units: int = 50, scaler=None, scalerMode="-", solver="adam",):
        self.hidden_layers = hidden_layers
        self.NrofHiddenLayers = len(hidden_layers)
        self.max_iters = max_iters
        self.random_state = random_state
        self.activation = actFunction
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver=solver,  activation=actFunction, max_iter=max_iters, random_state=random_state)
        self.scaler = scaler
        self.scalerMode = scalerMode
        self.predictions = []
        self.mse = 0
        self.weights = []
        self.rrmse = 0
        self.mape = 0
        self.diagram = None
        self.accuracy = 0
        self.modelStr = self.NrofHiddenLayers * '{}, '
        self.modelStr = "("+self.modelStr.format(*hidden_layers)[:-1]+")"
        self.x_test = []
        self.y_test = []
        self.x_train = []
        self.y_train = []


    def train_model(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.weights = [layer_weights for layer_weights in self.model.coefs_]
      #  self.r2_score = self.model.score(self.x_train, self.y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
    

    def forecastFutureValues(self, n, x_test, scaler=None, scalerMode="-", normOut = False):
        future_forecasts = []
        x_axis = []    
        input = x_test[-1].reshape(1, -1)  # Átalakítjuk a legutolsó  input értéket 2D formátumra

        for i in range(n):
            forecast = self.predict(input)[0]  # a predict 2d-s lisát ad vissza, 1 elemmel, mert csak 1 input van
            print(f"foreacast before: {forecast}")
                # ha a tanítás során nem voltak normalizálva az outputok de az inputok igen, 
                # akkor most utólag kell normalizálni a becslést, mielőtt az inputok közé tennénk, mivel nem lenne releváns pl 0.93, 0.93, 3.97
            if(normOut == False):
                if(scalerMode == "log"):
                    # elág csak a becslést logaritmizálni
                    forecast = np.log(np.longdouble(forecast))

            future_forecasts.append(forecast) 
            print(f"{i+1}. : {input} ---> {forecast}")
            x_axis.append(f"{i+1}. jóslat")

            #csúsztatjuk egyel arréb toljuk az input elmeit, az utolsó a legutóbbi előrejelzett érték lesz
            input = np.hstack((input[:, 1:], forecast.reshape(1, -1)))

            if(scaler is not None and not normOut):
              # az eredeti input skálázó példánnyal normlaizáljuk újra az egész input mintát, és abból kiveszem az átaslakított beccslést, majd csak azt cserélem ki, hogy ne változzon az előző két érték  
                last_value_normalized = scaler.transform(np.array(input).reshape(1, -1))[0][-1]           
                input[-1][-1] = last_value_normalized
                print(f"transzformált input: {input}")


        if(scalerMode == "log"):
            #a megjelenítés előtt visszatranszformáljuk rendes alaklba a  becsléseket
            future_forecasts = np.exp(future_forecasts).tolist()

        elif(scaler is not None and normOut) :
            future_forecasts = scaler.inverse_transform(np.array(future_forecasts).reshape(1, -1)).flatten().tolist()

        print(f"\n ****future firecast: \n x={x_axis}, \n y={future_forecasts}")
        return future_forecasts, x_axis
    
class Vanilla_LSTM:
    def __init__(self, learning_data, test_data, units:int = 50, activation: str = "relu", 
                  solver: str = "adam", scaler: str = "None", n_features: int = 1, n_steps: int = 3, input_dim: int = 100, loss: str ="mse",  epochs: int = 200, verbose: int = 0, n_pred:int = 6, name: str="default", normOut: bool = False):
        
        self.diagram = None
        self.epochs = epochs
        self.verbose = verbose
        self.activation = activation
        self.n_steps = n_steps
        self.n_features = n_features
        self.loss = loss
        self.units = units
        self.n_pred = n_pred
        self.scalerStr = scaler.strip()
        self.learning_data = learning_data
        self.test_data = test_data
        self.solver = solver
        self.name = name

        self.model = Sequential()
        self.model.add(LSTM(units=units, activation=activation, input_shape=(n_steps, n_features)))
        self.model.add(Dense(1))
        self.model.compile(optimizer=solver, loss=loss)
        
        print(f"\n ------------------creating {self.name}'s LSTM model and forecasts------------------/n")
        self.createMLsets()
        self.normalization(self.scalerStr, normOut)
        self.trainModel()
        self.predict()      
        self.kiertekel()
   
    
    def createMLsets(self):
        self.x_train, self.y_train = split_sequence(self.learning_data, self.n_steps)
        self.x_test, self.y_test = split_sequence(self.test_data, self.n_steps) 
        # reshape from [samples, timesteps] into [samples, timesteps, features] for LSTM 
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1], self.n_features)); self.x_test =  self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1], self.n_features))    

    def normalization(self, scaler, normalizeOutputs=False):
        self.scaler = None
        self.normalizeOutputs = normalizeOutputs

        if(scaler == "minmax"):
           self.scaler = MinMaxScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "standard":
            self.scaler = StandardScaler()
        elif self.scaler is None and scaler != "log":
            return

        if self.scaler is not None:
            self.x_train_Normalized = self.scaler.fit_transform(self.x_train.reshape(-1, self.x_train.shape[-1])).reshape(self.x_train.shape)
            self.x_test_Normalized = self.scaler.fit_transform(self.x_test.reshape(-1, self.x_test.shape[-1])).reshape(self.x_test.shape)

            if normalizeOutputs:
                self.y_train_Normalized = self.scaler.fit_transform(self.y_train.reshape(-1, 1))
                self.y_test_Normalized = self.scaler.fit_transform(self.y_test.reshape(-1, 1))

        if scaler == "log":
            self.x_train_Normalized = np.log(self.x_train, out=np.zeros_like(self.x_train), where=(self.x_train != 0))
            self.x_test_Normalized = np.log(self.x_test, out=np.zeros_like(self.x_test), where=(self.x_test != 0))
            
            if normalizeOutputs:
                self.y_train_Normalized = np.log(self.y_train).reshape(-1, 1)
                self.y_test_Normalized = np.log(self.y_test).reshape(-1, 1)
            
    def trainModel(self): 
        if self.scaler is not None or self.scalerStr == "log":
            if self.normalizeOutputs:
                self.model.fit(self.x_train_Normalized, self.y_train_Normalized, epochs=self.epochs, verbose=self.verbose)
            else:
                self.model.fit(self.x_train_Normalized, self.y_train, epochs=self.epochs, verbose=self.verbose)
        else:
            self.model.fit(self.x_train, self.y_train, epochs = self.epochs, verbose = self.verbose)
 
    def predict(self):
         # előrejelzés a tesztadatokra
        if self.scaler is not None or self.scalerStr == "log":
            self.predictions = self.model.predict(self.x_test_Normalized, verbose=self.verbose)

            #ha az outputok is normalizálva lettek, visszaállítás eredeti formára
            if self.normalizeOutputs:
                self.predictions_Normalized = self.predictions

                if self.scalerStr == "log":
                    self.predictions = np.exp(self.predictions)
                else:
                  self.predictionsNormalized = self.predictions
                  self.predictions = self.scaler.inverse_transform(self.predictions)
        else:
            self.predictions = self.model.predict(self.x_test, verbose=self.verbose)
            
        self.predictions = [round(item, 2) for sublist in self.predictions for item in sublist]

    def forecast_future(self):
        input = []
        forecasts = []

        if self.scaler is not None or self.scalerStr == "log":
            input = self.x_test_Normalized[-1].reshape(1, self.n_steps, self.n_features)
        else:
            input = self.x_test[-1].reshape(1, self.n_steps, self.n_features)

        print(f"prediction starts with {input} \n ")

        for i in range(self.n_pred):
                forecast =  self.model.predict(input, verbose = self.verbose)[0]
                print(f"{input} ==>  {forecast}")
                forecasts.append(forecast)
                #utolsó két elem előrecsúsztatása
                input = np.roll(input, -1, axis=1)
                
              
                if(not self.normalizeOutputs):
                    if self.scalerStr =="log":
                         last_value_normalized = np.log(forecast)
                         
                    if self.scaler is not None:
                        #a legelső input normalizálva van, viszont az előrejelzett érték nem a skálávan lesz, így át kell alakítani mielőtt beletesszük az inputba
                        forecast_array = np.array([forecast])
                        last_value_normalized = self.scaler.transform(forecast_array)[0][-1]   

                    #input frissitese az elorejelzett ertekkel  
                    else:
                        last_value_normalized = forecast      
                    input[-1][-1] = last_value_normalized
                else:
                    input[-1][-1] = forecast

        if(self.normalizeOutputs):
            #hogyha a kiement is egy logaritmizált érték, akkor a becsléseket vissza kell alakítani
            if self.scalerStr =="log":
               forecasts = np.exp(forecasts)

            elif self.scaler is not None:
            #ha a jóslatok is skálázott ér, azt vissza kell alakítani a becsléseket
                forecasts = np.array(forecasts).reshape(-1, 1)
                forecasts = self.scaler.inverse_transform(forecasts)
                             
        forecasts = [item for sublist in forecasts for item in sublist]
        print("\n \t\t\t DONE \n", forecasts)
        return forecasts


    def kiertekel(self):
        self.forecastZipped = zip(self.predictions, self.y_test)
        self.mse = MSE(self.predictions, self.y_test)
        self.rrmse = RRMSE(self.predictions, self.y_test)
        self.mape = MAPE(self.predictions, self.y_test)

   
    def printTraintSet(self):
        res = "<h1>training set: x == > y</h1>"

        if self.scaler is not None or self.scalerStr == "log":
            if self.normalizeOutputs:
                for i in range(len(self.x_train)) :
                    res += f"{i+1}.: {self.x_train_Normalized[i]} ==> {self.y_train_Normalized[i]} ) <br>"
                    res += f"____ {self.x_train[i]} ==> {self.y_train[i]}  <br><br>"
            else:
                for i in range(len(self.x_train)) :
                    res += f"{i+1}.: {self.x_train_Normalized[i]} ==> {self.y_train[i]} ) <br>"
                    res += f"___ {self.x_train[i]} ==> {self.y_train[i]}  <br><br>"
        else:
            for i in range(len(self.x_train)) :
                res += f"{i+1}.: {self.x_train[i]} ==> {self.y_train[i]} <br><br>"

        return res
    
    def printTestSet(self):
        res = f"<h1> prediction set: x (input) == > y </h1> <br>"

        if self.scaler is not None or self.scalerStr == "log":
            if self.normalizeOutputs:         
                for i in range(len(self.x_test_Normalized)) :
                    joslatNormalizalt = np.round(float(self.predictions_Normalized[i]), 2)
                    joslat = np.round(float(self.predictions[i]), 2)

                    res += f"{i+1}.: {self.x_test_Normalized[i]} ==> {self.y_test_Normalized[i]}, joslat: {joslatNormalizalt} <br>"
                    res += f"___ {self.x_test[i]} ==> {self.y_test[i]}, joslat: {joslat} <br><br>"
            else:
                for i in range(len(self.x_test_Normalized)) :
                    joslat = np.round(float(self.predictions[i]), 2)
                    res += f"{i+1}.: {self.x_test_Normalized[i]} ==> {self.y_test[i]} <br>"
                    res += f"___ {self.x_test[i]} ==> {self.y_test[i]}, joslat: {joslat} <br><br>"

        else:
            for i in range(len(self.x_test)) :
                joslat = np.round(self.predictions[i], 2)
                res += f"{i+1}.: {self.x_test[i]} ==> {self.y_test[i]}, joslat: {joslat} <br><br>"
        
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
        print("A becsült és valós értékek listáinak azonos hosszúnak kell lenniük.")
        return -1

    absolute_percentage_errors = []
    for prediction, actual in zip(becslesek, teszt_adatok):
        absolute_percentage_error = abs((actual - prediction) / actual) * 100
        absolute_percentage_errors.append(absolute_percentage_error)

    mean_absolute_percentage_error = sum(absolute_percentage_errors) / len(absolute_percentage_errors)
    return mean_absolute_percentage_error


import io
import base64
import numpy as np
import matplotlib.pyplot as plt

def plot_error_analysis(residuals, num_bins=10, name=""):
    # Minimum and maximum error values
    min_error, max_error = min(residuals), max(residuals)
    num_bins = int((len(residuals)) * 0.5)

    # Error histogram buffer
    hist_buffer = io.BytesIO()

    # Calculate histogram
    hist_values, bin_edges = np.histogram(residuals, bins=np.linspace(min_error, max_error, num_bins + 1))

    # Plot histogram
    plt.bar(bin_edges[:-1], hist_values, color='blue', edgecolor='black', width=bin_edges[1] - bin_edges[0])

    plt.xlabel('Tartomány')
    plt.ylabel('Gyakoriság')
    plt.title(name+' előrejelzési hibák eloszlása')

    # Set y-axis ticks to integer values
    plt.yticks(np.arange(0, max(hist_values) + 1, 1))

    plt.savefig(hist_buffer, format="png")
    hist_buffer.seek(0)
    encoded_hist_image = base64.b64encode(hist_buffer.getvalue()).decode('utf-8')
    plt.close()

    return encoded_hist_image

def plot_Residuals(residuals, name=""):
    # Lineáris illesztés
    x = np.arange(len(residuals))
    slope, intercept = np.polyfit(x, residuals, 1)
    line = slope * x + intercept
    # Reziduumok grafikon buffer
    residuals_buffer = io.BytesIO()
    # Reziduumok grafikon
    plt.plot(residuals, marker='o', linestyle='', color='blue')
    plt.plot(line, linestyle='-', color='red')
    plt.xlabel('Előrejelzés sorszáma')
    plt.ylabel('Reziduum')
    plt.title(name+' Előrejelzések reziduumai')
    # Y tengely intervallum beállítása
    plt.ylim(np.min(residuals) - 0.75, np.max(residuals) + 0.75)
    plt.savefig(residuals_buffer, format="png")
    residuals_buffer.seek(0)
    encoded_residuals_plot = base64.b64encode(residuals_buffer.getvalue()).decode('utf-8')
    plt.close()
    return encoded_residuals_plot


def Ljung_Box(residuals):
    lag = len(residuals) // 4
    result = sm.stats.diagnostic.acorr_ljungbox(residuals, lags=lag, return_df=True)
    p_values = [] 
    stats = []
    for i in range(lag):
        p_values.append(result.loc[i+1, 'lb_pvalue'])
        stats.append(result.loc[i+1, 'lb_stat'] )

    return stats, p_values

def White(residuals):
    """ Azt nézi, hogy a hibák varrianciája állandó-e azáltal, hogy homoszkedaszicitás vagy heteroszkedaszicitás van jelen.
        H0: Nincs Heteroszkedaszicitás (homoszkedaszicitás) ---> ez a jó, mert állandó  hibaszórás
        H1: Heteroscedasticity is present. --> nem jó
        Ha p > 0.05 nem utasítjuk el a nullhipotézist, tehát nincs jelen heteroszkedaszicitás, --> ez a jó
        Ha p < 0.05 akkor sajnos elutasítjuk H0-t, tehát a hibák varrianciája nem állandó"""
    squared_errors = np.square(residuals)
    exog = np.arange(len(squared_errors))
    exog = sm.add_constant(exog) 
    white_results = het_white(squared_errors, exog)

    p_value_homoskedasticity = white_results[1]
    p_value_heteroskedasticity = white_results[1]

    print(f"P-érték a homoszkedaszticitás teszthez: {p_value_homoskedasticity:.4f}")
    print(f"P-érték a heteroszkedaszticitás teszthez: {p_value_heteroskedasticity:.4f}")

    if p_value_heteroskedasticity < 0.05:
        res = f"White-teszt: <br> p = {round(p_value_heteroskedasticity, 2)} < 0.05  --> elutasítjuk H0-t. <br> A modell heteroszkedaszticitást mutat,  <br> tehát nem állandó a a hibák varrianciája, nem igazán megbízható."
    else:
        res = f"White-teszt: <br>  p = {round(p_value_heteroskedasticity,2)} > 0.05 --> nem utasítjuk H0-t. <br> A modell nem mutat heteroszkedaszticitást, <br> állandó a hibák varrianciája, megbízhatónak mondható."

    return res