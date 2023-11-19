import base64
from datetime import datetime
import os
import matplotlib.dates as mdates
from django.http import HttpResponse
from django.template import loader
import io
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .models import Stat
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from django.shortcuts import redirect, reverse


global statisztikak
statisztikak = []

def home(request):
    return render(request, 'home.html')

def statistics(request):
    if 'file' not in request.FILES:
        return HttpResponse("Nem lett fájl feltöltve!", status=400)

    uploaded_file = request.FILES['file']
    suruseg = int(request.POST['suruseg'])
    sheetName = request.POST['sheet']
    adatsorok = []     # kétdimenziós lista, az első oszlop utáni oszlopokat (megyék idősorait) tárolja
    adatsorNevek = []  # az oszlopok fejlécei, pl. a megyék nevei
    idoPontok = []     # a legelső oszlop, a megfigyelések időpontjait tárolja
    acfpacf = []       # acf és pacf tesztek diagrammjait tárolja képekben

    teszt_adatok =  request.FILES['file_teszt'] # az előrejelzett adatokkal fogjuk összehasonlítani
    tesztSheetName = request.POST['tesztSheetName']

    try:
        global df
        df = pd.read_excel(uploaded_file, sheet_name=sheetName)
        fejlec = df.columns.tolist()
        idoPontok = df[fejlec[0]].tolist()
        print(idoPontok)
        for i in range(len(fejlec)):
            if i > 0:
                adatsorNevek.append(fejlec[i])
                adatsorok.append(df[fejlec[i]].tolist())

        data_rows = []
        for i in range(len(idoPontok)):
            data_row = {'idoPont': idoPontok[i], 'adatsorok': [adatsor[i] for adatsor in adatsorok]}
            data_rows.append(data_row)
        
        diagram = AbrazolEgyben(adatsorok, idoPontok, adatsorNevek, suruseg)
        diagram = base64.b64encode(diagram.read()).decode('utf-8')

        global statisztikak
        statisztikak = Statisztikak(adatsorNevek, adatsorok, idoPontok)

        for i in range(len(adatsorNevek)):
            plot = (plot_acf_and_pacf(adatsorok[i], adatsorNevek[i]))
            acfpacf.append( base64.b64encode(plot.read()).decode('utf-8'))

        
        teszt_adatok_df = pd.read_excel(teszt_adatok, sheet_name=tesztSheetName)
        for i in statisztikak:
            for j in fejlec:
                if(i.megye_nev == j) :
                   i.setTesztAdatok(teszt_adatok_df[j].tolist())
            print(i.teszt_adatok)
        

        return render(request, 'showData.html', {'data_rows': data_rows, 'adatsorNevek': adatsorNevek, 'statisztikak':statisztikak, 'diagram': diagram, 'acfpacf': acfpacf })

    except pd.errors.ParserError:
        return HttpResponse("Helytelen fájl!", status=400)


def AbrazolEgyben(adatok, idoszakok, megyek, suruseg): 
    utolso_ev_ho = idoszakok[-1] 
    elso_ev_ho = idoszakok[0]
    plt.figure(figsize=(15, 7))
    for i, megye in enumerate(megyek): 
        plt.plot(idoszakok, adatok[i], label=megye)

    plt.xlabel('Időszak')
    plt.ylabel('Munkanélküliségi ráta (%)')
    plt.title(f"Székelyföldi megyék Munkanélküliségi rátái {elso_ev_ho} - {utolso_ev_ho} között")
    plt.grid(True)
    try:
      plt.xticks(idoszakok[::suruseg], rotation=45, ha="right", fontsize=8)
    except:
        pass
    plt.legend()
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)  
    return buffer
    

def SzamitStatisztikak(adatok: list):
    statisztikak = {
        "átlag": round(np.mean(adatok),4),
        "szórás": round(np.std(adatok),4),
        "variancia": round(np.var(adatok),4),
        "medián": round(np.median(adatok), 4),
        "min": np.min(adatok),
        "max": np.max(adatok)
    }
    return statisztikak

def Statisztikak(megyek, adatok, idoPontok):
    eredmenyek = []
    for i in range(len(megyek)):
        statisztika = Stat(megyek[i], adatok[i], idoPontok)
        eredmenyek.append(statisztika)
    return eredmenyek


def plot_acf_and_pacf(data, megye_nev):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True) 
    fig.subplots_adjust(hspace=0.3)  
    plot_acf(data, lags=40, ax=ax1, title=f"Autokorreláció ({megye_nev})")
    plot_pacf(data, lags=40, ax=ax2, title=f"Parciális Autokorreláció ({megye_nev})")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)  
    return buffer



def arima(request):
    if not statisztikak:
        return HttpResponse("Statisztikak is not initialized", status=400)

    model_summary_file_path = 'model_summary.txt'
    forecast_file_path = 'arima_forecasts.txt'

    forecasts = {}  # Inicializáld a forecasts szótárat a ciklus előtt

    # Nyisd meg a fájlokat egyszer a ciklus előtt
    with open(model_summary_file_path, 'w+') as model_summary_file, open(forecast_file_path, 'w+') as forecast_file:
        t = int(request.POST['t'])
        idoszakok_ = ["2022-10", "2022-11", "2022-12", "2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06", "2023-07"]
        megyek = []
        adatsorok =[] 

        for megye in statisztikak:
            p = request.POST[megye.megye_nev+'_p']
            q = request.POST[megye.megye_nev+'_q']
            d = request.POST[megye.megye_nev+'_d']
            megye.teszt_idoszakok = idoszakok_ 
            tipus = request.POST[megye.megye_nev+'_tipus']
            test_results = None
            title = None


            if tipus == "ar":
                test_results = megye.AR(p, t)
                title = f"\n{megye.megye_nev} AR({p})\n"

            elif tipus == "ma":
                test_results = megye.MA(q, t)
                title = f"\n{megye.megye_nev} MA({q})\n"

            elif tipus == "arma":
                test_results = megye.ARMA(p, q, t)
                title = f"\n{megye.megye_nev} ARMA({p}, {q})\n"

            if test_results:
                model_summary_file.write(f"{'='*40}\n{title}{'='*40}\n")
                model_summary_file.write(str(test_results[0]))
                model_summary_file.write('\n\n')
                model_summary_file.write(str(test_results[1]))
                megye.model = title
                forecast_file.write('\n\n')
                megyek.append(megye.megye_nev)
                adatsorok.append(megye.becslesek)


    diagram = AbrazolEgyben(adatsorok, idoszakok_, megyek, 1)
    diagram = base64.b64encode(diagram.read()).decode('utf-8')
    
    return render(request, "arimaForecasts.html", {"megyek": statisztikak, "file": model_summary_file_path, "diagram": diagram})


def download(request):
    file_path = 'model_summary.txt'  # Az aktuális fájl elérési útvonala
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
        return response

def mse(request):
    return render(request, "arimaForecasts.html")