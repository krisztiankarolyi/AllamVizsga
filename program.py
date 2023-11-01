import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


filename = "C:/Users/Károlyi Krisztián/Desktop\SAPI_3-1/AllamVizsga/data.xlsx"
mnk_rata_cv = []; mnk_rata_hr = []; mnk_rata_ms = []; idoszakok  = []
adatokSzama = int(input("Hány db megfigyelés van az adatsorban összesen?"))

def Beolvas(filename: str, evek: int):
    global mnk_rata_cv, mnk_rata_hr, mnk_rata_ms, idoszakok, adatokSzama
    kezdosor =  int(adatokSzama - (12*evek)-1)
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

def AbrazolEgyben(adatok, idoszakok, megyek, suruseg):
    utolso_ev_ho = idoszakok[-1]  # Az utolsó év és hónap meghatározása
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


# Autokorrelációs és parciális autokorrelációs tesztek
def plot_acf_and_pacf(data, megye_nev):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    plot_acf(data, lags=40, ax=ax1, title=f"Autokorreláció ({megye_nev})")
    plot_pacf(data, lags=40, ax=ax2, title=f"Parciális Autokorreláció ({megye_nev})")


adatokSzama = 379 
evek = int(input("Hány évre visszamenőleg dolgozzam fel az adatsort?"))
suruseg = int(input("Milyen sűrűséggel legyen az X tengely?"))
Beolvas(filename, evek)                               # hány évre visszamenőleg kezdje el beolvasni
adatok = [mnk_rata_cv, mnk_rata_hr, mnk_rata_ms] #egyberakom a három megye adatait, hogy dinamikusabban hívhassam a függvényeket
megyek = ["CV", "HR", "MS"]                      #segít megjelölni hogy az adatok listában melyik adatsor melyik megyét jelenti
Kiir(megyek, adatok, idoszakok)
statisztikak = Statisztikak(megyek, adatok)
cvAtlag = GetStatisztika(statisztikak, "CV", "átlag")
ShowStatisztikak(statisztikak)
print("Kovászna átlaga: ", cvAtlag)
AbrazolEgyben(adatok, idoszakok, megyek, suruseg)

# Autokorrelációs és parciális autokorrelációs tesztek
for i, megye in enumerate(megyek):
    plot_acf_and_pacf(adatok[i], megye)

plt.tight_layout()
plt.show()

