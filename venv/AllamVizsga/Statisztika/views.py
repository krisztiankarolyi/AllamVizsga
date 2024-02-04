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

        for megye in statisztikak:
            megye.plot_acf_and_pacf()

        teszt_adatok_df = pd.read_excel(teszt_adatok, sheet_name=tesztSheetName)
        for i in statisztikak:
            for j in fejlec:
                if i.megye_nev == j:
                    i.setTesztAdatok(teszt_adatok_df[j].tolist())

        return render(request, 'showData.html', {'data_rows': data_rows, 'adatsorNevek': adatsorNevek, 'statisztikak': statisztikak, 'diagram': diagram})

    except pd.errors.ParserError:
        print(traceback.format_exc())
        return HttpResponse("Helytelen fájl!", status=400)


import traceback
import io
import matplotlib.pyplot as plt
import numpy as np

def AbrazolEgyben(adatsorok, idoszakok, megnevezesek, suruseg, Cim="", yFelirat="", y_min=None, y_max=None, y_step=None): 
    plt.figure(figsize=(15, 7))
    
    for i, megye in enumerate(megnevezesek): 
        plt.plot(idoszakok, adatsorok[i], label=megye)

    plt.ylabel(yFelirat)
    plt.title(f"{Cim} {idoszakok[0]} - {idoszakok[-1]} között")
    plt.grid(True)

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
        model_summary_file_path = 'model_summary.txt'
        forecast_file_path = 'arima_forecasts.txt'

        with open(model_summary_file_path, 'w+') as model_summary_file, open(forecast_file_path, 'w+') as forecast_file:
            global beolvasott_teszt_idoszakok
            megyek = []
            adatsorok =[] 
            eredeti_adatsorok = []

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
                    model_summary_file.write(f"{'='*40}\n{title}{'='*40}\n")
                    model_summary_file.write(str(test_results[0]))
                    model_summary_file.write('\n\n')
                    model_summary_file.write("Elorejelzett ertekek: "+str(test_results[1]))
                    megye.model = title
                    model_summary_file.write('\n\n')
                    model_summary_file.write('MSE: '+str(megye.MSE())+ "\nRMSE: "+ str(megye.RRMSE())+"\n")

                    megyek.append(megye.megye_nev)
                    adatsorok.append(megye.becslesek)
                    eredeti_adatsorok.append(megye.teszt_adatok)
        

        diagaram_teszt = AbrazolEgyben(adatsorok, beolvasott_teszt_idoszakok, megyek, 1, "Székelyföld előrejelzett munkanélküliségi rátái", "", 2, 5, 0.5)
        diagaram_teszt = base64.b64encode(diagaram_teszt.read()).decode('utf-8')

        diagram_eredeti = AbrazolEgyben(eredeti_adatsorok, beolvasott_teszt_idoszakok, megyek, 1, "Székelyföld mért munkanélküliségi rátái", "", 2, 5, 0.5)
        diagram_eredeti = base64.b64encode(diagram_eredeti.read()).decode('utf-8')
            
        return render(request, "arimaForecasts.html", {"megyek": statisztikak, "file": model_summary_file_path, "diagaram_teszt": diagaram_teszt, "diagram_eredeti": diagram_eredeti})
    
    except:
        print(traceback.format_exc())
        return redirect('home')

def download(request):
    file_path = 'model_summary.txt'
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/force-download')
        response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
        return response

