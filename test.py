import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st


filename = "C:/Users/Károlyi Krisztián/Desktop\SAPI_3-1/AllamVizsga/data.xlsx"
mnk_rata_cv = []; mnk_rata_hr = []; mnk_rata_ms = []; idoszakok  = []
adatokSzama = 379 

def Beolvas(filename: str, evek: int):
    global mnk_rata_cv, mnk_rata_hr, mnk_rata_ms, idoszakok, adatokSzama
    kezdosor =  adatokSzama - 12*evek
    data = pd.read_excel(filename, sheet_name='data')
    mnk_rata_cv = data['mnk_rata_cv'].tolist()[kezdosor:]
    mnk_rata_hr = data['mnk_rata_hr'].tolist()[kezdosor:]
    mnk_rata_ms = data['mnk_rata_ms'].tolist()[kezdosor:]
    idoszakok = data['idoszak'].tolist()[kezdosor:]

def Kiir(megyek: list, adatok: list, idoszakok: list):
    print("Adatok (mért hónapok) száma : "+str(len(idoszakok)))
    print("Időszak:" + "\t\t " + "\t".join(megyek))  # Kiírjuk a megyék neveit fejlécként
    print("-----------------------------------------------")
    for i in range(len(adatok[0])):  # Feltételezzük, hogy az első megye adatsorának hossza minden megyére vonatkozik 
        datum = idoszakok[i]+"\t"
        sor = [datum] + [str(adat[i]) for adat in adatok]  # Sor létrehozása az összes megye adataiból
        print("\t".join(sor))


def AbrazolKulon(adatok: list, idoszakok: list, megye: str):
    plt.figure(figsize=(15, 7))  
    label = "Munkanélküliségi ráta százalékos értékei havonta " + megye + " megyében";
    plt.plot(idoszakok, adatok, label=label) 
    plt.xlabel('Időszak')  
    plt.ylabel('Munkanélküliségi ráta (%)')  
    plt.title(label)  
    plt.grid(True)  
    plt.xticks(idoszakok[::12], rotation=90, fontsize=8)  
    plt.legend()  
    plt.show()

def AbrazolEgyben(adatok: list, idoszakok: list, megyek: list, suruseg):
    plt.figure(figsize=(15, 7))
    for i, megye in enumerate(megyek):
        plt.plot(idoszakok, adatok[i], label=megye)
    plt.xlabel('Időszak')
    plt.ylabel('Munkanélküliségi ráta (%)')
    plt.title("Székelyföldi megyék Munkanélküliségi rátái 1992 január - 2023 július")
    plt.grid(True)
    plt.xticks(idoszakok[::suruseg], rotation=90, fontsize=8)
    plt.legend()
    plt.show()

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

def Statisztikak(megyek: list, adatok: list):
    eredmenyek = []
    for i, megye in enumerate(megyek):
        statisztikak = SzamitStatisztikak(adatok[i])
        eredmenyek.append({megye: statisztikak})
    return eredmenyek

def ShowStatisztikak(statisztikak: list):
    for eredmeny in statisztikak:
        megye, stat = eredmeny.popitem()
        print(f"Megye: {megye}")
        for kulcs, ertek in stat.items():
            print(f"\t{kulcs}: {ertek:.2f} %")  # Két tizedesjegyre formázza az értékeket
        print()

def GetStatisztika(statisztikak, megye, adat_neve):
    for megye_stat in statisztikak:
        megye_stat_copy = copy.deepcopy(megye_stat)  # Create a deep copy of the dictionary
        megye_nev, stat = megye_stat_copy.popitem()
        if megye_nev == megye and adat_neve in stat:
            return stat[adat_neve]
    return None

adatokSzama = 379 
Beolvas(filename, 5)  # hány évre visszamenőleg kezdje el beolvasni
adatok = [mnk_rata_cv, mnk_rata_hr, mnk_rata_ms] #egyberakom a három megye adatait, hogy dinamikusabban hívhassam a függvényeket
megyek = ["CV", "HR", "MS"]                      #segít megjelölni hogy az adatok listában melyik adatsor melyik megyét jelenti
Kiir(megyek, adatok, idoszakok)
statisztikak = Statisztikak(megyek, adatok)
cvAtlag = GetStatisztika(statisztikak, "CV", "átlag")
ShowStatisztikak(statisztikak)
print("Kovászna átlaga: ", cvAtlag)
#AbrazolEgyben(adatok, idoszakok, megyek, 1)
