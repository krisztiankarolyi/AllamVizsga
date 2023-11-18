import base64
import os
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
# ...

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

    try:
        global df
        df = pd.read_excel(uploaded_file, sheet_name=sheetName)
        fejlec = df.columns.tolist()
        idoPontok = df[fejlec[0]].tolist()

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
    plt.xticks(idoszakok[::suruseg], rotation=90, fontsize=8)
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

    resp = []
    model_summary_file_path = 'model_summary.txt'
    forecast_file_path = 'arima_forecasts.txt'

    # Nyisd meg a fájlokat egyszer a ciklus előtt
    with open(model_summary_file_path, 'w+') as model_summary_file, open(forecast_file_path, 'w+') as forecast_file:
        for megye in statisztikak:
            p = request.POST[megye.megye_nev+'_p']
            q = request.POST[megye.megye_nev+'_q']
            d = request.POST[megye.megye_nev+'_d']
            tipus = request.POST[megye.megye_nev+'_tipus']

            if(tipus == "ar"):
                test_results = megye.AR(p)
                model_summary_file.write(f"{'='*40}\n{megye.megye_nev} AR({p})\n{'='*40}\n")
                model_summary_file.write(str(test_results[0])) 
                model_summary_file.write('\n\n')
                forecast_file.write(str(test_results[1])) 
                forecast_file.write('\n\n')
                title = "\n" + megye.megye_nev + " AR(" + p + ")\n"
                resp.append(title + str(test_results[0]))

            if(tipus == "ma"):
                test_results = megye.MA(q)
                model_summary_file.write(f"{'='*40}\n{megye.megye_nev} MA({q})\n{'='*40}\n")
                model_summary_file.write(str(test_results[0])) 
                model_summary_file.write('\n\n')
                forecast_file.write(str(test_results[1])) 
                forecast_file.write('\n\n')
                title = "\n" + megye.megye_nev + " MA(" + q + ")\n"
                resp.append(title + str(test_results[0]))

            if(tipus == "arma"):
                test_results = megye.ARMA(p, q)
                model_summary_file.write(f"{'='*40}\n{megye.megye_nev} ARMA({p}, {q})\n{'='*40}\n")
                model_summary_file.write(str(test_results[0])) 
                forecast_file.write(str(test_results[1]))  
                forecast_file.write('\n\n')
                title = "\n" + megye.megye_nev + " ARMA(" +p+", "+ q + ")\n"
                resp.append(title + str(test_results[0]))

    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = f'attachment; filename="model_summary.txt"'

    with open(model_summary_file_path, 'r') as model_summary_file:
        for line in model_summary_file:
            response.write(line)

    response.write('\nElőrejelzések\n')  # Add a separator between the two files

    with open(forecast_file_path, 'r') as forecast_file:
        for line in forecast_file:
            response.write(line)

    os.remove(model_summary_file_path)
    os.remove(forecast_file_path)

    return response




def arimaForecasts(request):
    return render(request, "arimaForecasts.html")

