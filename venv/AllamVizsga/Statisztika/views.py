import base64
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
    # Itt lehet bármilyen adatot előkészíteni a template számára, majd megjeleníteni
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)  # Hozz létre egy közös X-tengelyt
    fig.subplots_adjust(hspace=0.3)  # Állítsd be a térközt a két aldiagram között
    plot_acf(data, lags=40, ax=ax1, title=f"Autokorreláció ({megye_nev})")
    plot_pacf(data, lags=40, ax=ax2, title=f"Parciális Autokorreláció ({megye_nev})")
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)  
    return buffer


def arima(request):
    if not statisztikak:
        return HttpResponse("Statisztikak is not initialized", status=400)

    p = request.POST['p']; q = request.POST['q']; d = request.POST['d']
    tesztek = request.POST.getlist('teszt')
    resp = []

    if "ar" in tesztek:
        for i in range(len(statisztikak)):
            test = str(statisztikak[i].AR(p))
            title = "\n"+statisztikak[i].megye_nev + " AR(" + p + ")\n"
            resp.append(title + test)

    if "ma" in tesztek:
        for i in range(len(statisztikak)):
            test = str(statisztikak[i].MA(q))
            title = "\n"+statisztikak[i].megye_nev + " MA(" + q + ")\n"
            resp.append(title + test)

    if "arma" in tesztek:
        for i in range(len(statisztikak)):
            test = str(statisztikak[i].ARMA(p, q))
            title = "\n"+statisztikak[i].megye_nev + " ARMA(" + p + ", " + q + ")\n"
            resp.append(title + test)

    response = HttpResponse(content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename="arima_results.txt"'

    for line in resp:
        response.write(line)

    return response
