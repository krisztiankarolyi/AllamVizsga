import base64
from datetime import datetime
import os
import traceback
import matplotlib.dates as mdates
from django.http import HttpResponse
from django.template import loader
import io
from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from .models import Stat
import statsmodels.api as sm
from django.contrib import messages
from django.shortcuts import redirect

global statisztikak 

def home(request):
    messages.error(request, 'Nem lett adatforrás fájl feltöltve!')
    return render(request, 'home.html')

def upload(request):
    if 'file' not in request.FILES or 'suruseg' not in request.POST or 'sheet' not in request.POST:
        messages.error(request, 'Hiányzó paraméter(ek) (sűrűség/munkalap nevek)!')
        return redirect('home')

    uploaded_file = request.FILES['file']
    suruseg = int(request.POST['suruseg'])
    sheetName = request.POST['sheet']

    adatsorok, adatsorNevek, idoPontok, teszt_adatok = [], [], [], None

    if 'sameFile' in request.POST:
        teszt_adatok = request.FILES['file']

    elif 'file_teszt' not in request.FILES:
        messages.error(request, 'Nem lett adatforrás fájl feltöltve az előrjelzésekhez!')
        return redirect('home')
    
    else:
        teszt_adatok = request.FILES['file_teszt']

    tesztSheetName = request.POST['tesztSheetName']

    try:
        df_teszt = pd.read_excel(teszt_adatok, sheet_name=tesztSheetName)
        global beolvasott_teszt_idoszakok
        beolvasott_teszt_idoszakok = df_teszt[df_teszt.columns[0]].tolist()

        df = pd.read_excel(uploaded_file, sheet_name=sheetName)
        fejlec = df.columns.tolist()
        idoPontok = df[fejlec[0]].tolist()

        for i, col in enumerate(fejlec[1:]):
            adatsorNevek.append(col)
            adatsorok.append(df[col].tolist())

        data_rows = [{'idoPont': ido, 'adatsorok': [adatsor[i] for adatsor in adatsorok]} for i, ido in enumerate(idoPontok)]
        
        diagram = AbrazolEgyben(adatsorok, idoPontok, adatsorNevek, suruseg, "Székelyföld munkanélküliségi rátái", "")
        diagram = base64.b64encode(diagram.read()).decode('utf-8')

        global statisztikak
        statisztikak = createStatObjects(adatsorNevek, adatsorok, idoPontok)

        teszt_adatok_df = pd.read_excel(teszt_adatok, sheet_name=tesztSheetName)
        for i in statisztikak:
            for j in fejlec:
                if i.megye_nev == j:
                    i.setTesztAdatok(teszt_adatok_df[j].tolist()) 
                    i.setTesztIdoszakok(beolvasott_teszt_idoszakok)

        return render(request, 'upload.html', {'data_rows': data_rows, 'adatsorNevek': adatsorNevek, 'statisztikak': statisztikak, 'diagram': diagram})
    
    except pd.errors.ParserError:
        print(traceback.format_exc())
        return HttpResponse("Helytelen fájl!", status=400)



def BoxJenkins(request):
    global statisztikak
    for megye in statisztikak:
        megye.calculateStatistics()
        megye.plot_acf_and_pacf()
    return render(request, 'Box-Jenkins.html', {'statisztikak': statisztikak})

def MLP(request):
    try: 
        global statisztikak
        return render(request, 'MLP.html', {'statisztikak': statisztikak})
    except Exception as e:
        print(traceback.format_exc())
        return HttpResponse("Hiba történt. "+str(e))

def MLPResults(request):
    try:
        global statisztikak
        for megye in statisztikak:           
            scaler =  request.POST[megye.megye_nev+'_scaler']
            actFunction = request.POST[megye.megye_nev+'_actFunction']
            maxIters = request.POST[megye.megye_nev+'_max_iters']
            targetRRMSE = float(request.POST[megye.megye_nev+'_targetRRMSE'])/100
            solver = request.POST[megye.megye_nev+"_solver"]
            x_mode = request.POST[megye.megye_nev+"_x_mode"]
            n_delays = int(request.POST[megye.megye_nev+"_n_delay"])
            randomStateMin = int(request.POST[megye.megye_nev+'_random_state_min'])
            randomStateMax = int(request.POST[megye.megye_nev+'_random_state_max'])
            hidden_layers = tuple(map(int, request.POST[megye.megye_nev+'_hidden_layers'].split(',')))
            megye.predict_with_mlp(actFunction, hidden_layers, int(maxIters), scaler=scaler, randomStateMax=randomStateMax, randomStateMin=randomStateMin, solver=solver, targetRRMSE=targetRRMSE, x_mode=x_mode, n_delays = n_delays) 
            diagram = AbrazolEgyben([megye.mlp_model.predictions, megye.teszt_adatok], megye.teszt_idoszakok, [megye.megye_nev+" MLP", megye.megye_nev+" mért"], 1, megye.megye_nev+" megye előrejelzett munkanélküliségi rátái", "", 2, 5, 0.5)
            diagram = base64.b64encode(diagram.read()).decode('utf-8')
            megye.setMLPDiagram(diagram)

        adatsorNevek = []
        adatsorok = []

        for megye in statisztikak:
            adatsorNevek.append(megye.megye_nev)
            adatsorok.append(megye.teszt_adatok)
            adatsorNevek.append(megye.megye_nev+" MLP")
            adatsorok.append(megye.mlp_model.predictions)

        diagaramEgyben = AbrazolEgyben(adatsorok, beolvasott_teszt_idoszakok, adatsorNevek, 1, "Székelyföld előrejelzett munkanélküliségi rátái", "", 2, 5, 0.5)
        diagaramEgyben = base64.b64encode(diagaramEgyben.read()).decode('utf-8')

        return render(request, 'MLPForecasts.html', {'statisztikak': statisztikak, 'diagramEgyben': diagaramEgyben})
        
    except Exception as e:
        print(traceback.format_exc())
        return HttpResponse("Hiba történt. "+str(e))
    
def AbrazolEgyben(adatsorok, idoszakok, megnevezesek, suruseg, Cim="", yFelirat="", y_min=None, y_max=None, y_step=None, grid=False): 
    plt.figure(figsize=(15, 7))
    
    for i, megye in enumerate(megnevezesek): 
        plt.plot(idoszakok, adatsorok[i], label=megye)

    plt.ylabel(yFelirat)
    plt.title(f"{Cim} {idoszakok[0]} - {idoszakok[-1]} között")
    plt.grid(grid)

    try:
        plt.xticks(idoszakok[::suruseg], rotation=45, ha="right", fontsize=8)
    except Exception as e:
        print(f"Error: {e}")

    if all((y_min, y_max, y_step)):
        plt.yticks(np.arange(y_min, y_max, y_step))
    
    plt.legend()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)  
    return buffer

def createStatObjects(megyek, adatok, idoPontok):
    eredmenyek = []
    for i in range(len(megyek)):
        statisztika = Stat(megyek[i], adatok[i], idoPontok)
        eredmenyek.append(statisztika)
    return eredmenyek

def arima(request):
    try: 
        global beolvasott_teszt_idoszakok
        megyek = []
        adatsorok =[] 

        for megye in statisztikak:
            p = request.POST[megye.megye_nev+'_p']
            q = request.POST[megye.megye_nev+'_q']
            d = request.POST[megye.megye_nev+'_d']
            megye.teszt_idoszakok = beolvasott_teszt_idoszakok 
            tipus = request.POST[megye.megye_nev+'_tipus']
            test_results = None
            title = None
            t = len(beolvasott_teszt_idoszakok)

            if tipus == "ar":
                test_results = megye.AR(p, t)
                title = f"\n{megye.megye_nev} AR({p})\n"

            elif tipus == "ma":
                test_results = megye.MA(q, t)
                title = f"\n{megye.megye_nev} MA({q})\n"

            elif tipus == "arma":
                test_results = megye.ARMA(p, q, t)
                title = f"\n{megye.megye_nev} ARMA({p}, {q})\n"
            
            elif tipus == "arima":
                test_results = megye.ARIMA(p, d, q, t)
                title = f"\n{megye.megye_nev} ARIMA({p}, {d}, {q})\n"

            if test_results:
                megye.model = title
                megyek.append(megye.megye_nev)
                adatsorok.append(megye.ARIMAbecslesek)
                diagram = AbrazolEgyben([megye.ARIMAbecslesek, megye.teszt_adatok], megye.teszt_idoszakok, [megye.model, megye.megye_nev+" mért"], 1, megye.megye_nev+" megye előrejelzett munkanélküliségi rátái", "", 2, 5, 0.5)
                diagram = base64.b64encode(diagram.read()).decode('utf-8')
                megye.setARIMADiagram(diagram)
    
        adatsorNevek = []
        for megye in statisztikak:
            adatsorNevek.append(megye.megye_nev)
            adatsorok.append(megye.teszt_adatok)
            adatsorNevek.append(megye.model)
            adatsorok.append(megye.ARIMAbecslesek)

        diagaramEgyben = AbrazolEgyben(adatsorok, beolvasott_teszt_idoszakok, adatsorNevek, 1, "Székelyföld előrejelzett munkanélküliségi rátái", "", 2, 5, 0.5)
        diagaramEgyben = base64.b64encode(diagaramEgyben.read()).decode('utf-8')

        return render(request, "arimaForecasts.html", {"statisztikak": statisztikak, "diagaramEgyben": diagaramEgyben})

    except:
        print(traceback.format_exc())
        return redirect('home')


