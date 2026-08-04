import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, util
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy import ndimage as ndi

# SQLAlchemy für das ORM
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import declarative_base, sessionmaker

# --- 1. ORM DEFINITION (Datenbank-Modell) ---
Base = declarative_base()

class PorenAnalyseEintrag(Base):
    __tablename__ = 'poren_analysen'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bild_name = Column(String, unique=True)
    porositaet_px = Column(Float)
    poren_anzahl = Column(Integer)
    mittlerer_poren_durchmesser_px = Column(Float)
    median_poren_durchmesser_px = Column(Float)
    std_poren_durchmesser_px = Column(Float)
    mittlere_poren_kreisform = Column(Float)
    mittleres_poren_seitenverhaeltnis = Column(Float)
    skelett_laengen_dichte = Column(Float)
    skelett_verzweigungspunkte = Column(Integer)
    skelett_endpunkte = Column(Integer)
    fraktale_dimension = Column(Float)
    fehler = Column(String)

# SQLite-Datenbank initialisieren (lokale Datei)
engine = create_engine('sqlite:///poren_analyse.db', echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- 2. BILDVERARBEITUNGS-FUNKTIONEN ---
bild_dateien = [
    "/mnt/data/20230831_11p_17.tif",
    "/mnt/data/20230824_15p_18.tif",
    "/mnt/data/20230824_14p_18.tif",
    "/mnt/data/20230823_13p_17.tif",
    "/mnt/data/20230823_10p_17.tif",
    "/mnt/data/20230822_12p_21.tif"
]

def imread_grau_8bit(pfad: str):
    img = io.imread(pfad)
    if img.ndim == 3:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

def kontrast_verstaerken(img): 
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)

def entrauschen(img): 
    return cv2.bilateralFilter(img, 5, 10, 10)

def poren_binarisieren(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    komb = ((otsu > 0).astype(np.uint8) + (ad > 0).astype(np.uint8)) >= 1
    return komb if 0.2 <= komb.mean() <= 0.95 else (~komb)

def binarium_bereinigen(bin_img): 
    return remove_small_holes(remove_small_objects(bin_img, 25), 25)

def form_deskriptoren_berechnen(regions):
    kreisform_liste, seitenverhaeltnis_liste, aequivalente_durchmesser = [], [], []
    for r in regions:
        flaeche = r.area
        umfang = r.perimeter if r.perimeter > 0 else 1.0
        kreisform = 4.0 * math.pi * flaeche / (umfang ** 2)
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        seitenverhaeltnis = (max(h, w) / (min(h, w) + 1e-6))
        
        kreisform_liste.append(kreisform)
        seitenverhaeltnis_liste.append(seitenverhaeltnis)
        aequivalente_durchmesser.append(r.equivalent_diameter)
        
    if not regions:
        return np.nan, np.nan, []
        
    return np.nanmean(kreisform_liste), np.nanmean(seitenverhaeltnis_liste), aequivalente_durchmesser

def distanz_wasserscheide(binaer_maske):
    distanz = ndi.distance_transform_edt(binaer_maske)
    lokale_maxima = morphology.local_maxima(distanz)
    marker, _ = ndi.label(lokale_maxima)
    return morphology.watershed(-distanz, marker, mask=binaer_maske)

def skelett_metriken(feste_maske):
    skel = skeletonize(feste_maske)
    laenge_px = skel.sum()
    flaeche_px = skel.size
    dichte = laenge_px / flaeche_px
    
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]], dtype=np.uint8)
    nachbarn_filter = cv2.filter2D(skel.astype(np.uint8), -1, kernel) - 10
    
    verzweigungen = int(((nachbarn_filter >= 3) & skel).sum())
    endpunkte = int(((nachbarn_filter == 1) & skel).sum())
    return dichte, verzweigungen, endpunkte

def fraktale_dimension(binaer_maske):
    img = util.img_as_ubyte(binaer_maske > 0)
    groessen, anzahlen = [], []
    h, w = img.shape
    s = 1
    while s <= min(h, w):
        zugeschnitten = img[:(h // s) * s, :(w // s) * s]
        ansicht = zugeschnitten.reshape((h // s, s, w // s, s))
        block_summe = ansicht.sum(axis=(1,3))
        anzahl = np.count_nonzero(block_summe)
        if anzahl > 0:
            groessen.append(1.0 / s)
            anzahlen.append(anzahl)
        s *= 2
    if len(groessen) < 2: 
        return np.nan
    return float(np.polyfit(np.log(groessen), np.log(anzahlen), 1)[0])

# --- 3. HAUPTDURCHLAUF & ORM SPEICHERUNG ---
session = Session()

for pfad in bild_dateien:
    bild_name = os.path.basename(pfad)
    try:
        img = imread_grau_8bit(pfad)
        proc = entrauschen(kontrast_verstaerken(img))
        poren = binarium_bereinigen(poren_binarisieren(proc))
        labels = distanz_wasserscheide(poren)
        regions = measure.regionprops(labels)
        
        kreis, sv, aeq_durchmesser = form_deskriptoren_berechnen(regions)
        porositaet = poren.mean()
        skel_dichte, verzweigung, ende = skelett_metriken(~poren)
        fd = fraktale_dimension(poren)
        
        # Als ORM-Objekt erstellen/aktualisieren
        eintrag = session.query(PorenAnalyseEintrag).filter_by(bild_name=bild_name).first()
        if not eintrag:
            eintrag = PorenAnalyseEintrag(bild_name=bild_name)
            session.add(eintrag)
            
        eintrag.porositaet_px = porositaet
        eintrag.poren_anzahl = len(regions)
        eintrag.mittlerer_poren_durchmesser_px = np.nanmean(aeq_durchmesser) if aeq_durchmesser else np.nan
        eintrag.median_poren_durchmesser_px = np.nanmedian(aeq_durchmesser) if aeq_durchmesser else np.nan
        eintrag.std_poren_durchmesser_px = np.nanstd(aeq_durchmesser) if aeq_durchmesser else np.nan
        eintrag.mittlere_poren_kreisform = kreis
        eintrag.mittleres_poren_seitenverhaeltnis = sv
        eintrag.skelett_laengen_dichte = skel_dichte
        eintrag.skelett_verzweigungspunkte = verzweigung
        eintrag.skelett_endpunkte = ende
        eintrag.fraktale_dimension = fd
        eintrag.fehler = ""
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Fehler bei {bild_name}: {e}")

# --- 4. DATEN AUS ORM LADEN & MULTIPLOT ERZEUGEN ---
# Wir holen alle Analysen direkt aus der SQLite-Datenbank via SQLAlchemy ORM
ergebnisse = session.query(PorenAnalyseEintrag).all()

noms = [e.bild_name for e in ergebnisse]
porositaeten = [e.porositaet_px for e in ergebnisse]
poren_anzahlen = [e.poren_anzahl for e in ergebnisse]
durchmesser = [e.mittlerer_poren_durchmesser_px for e in ergebnisse]
fraktal_dims = [e.fraktale_dimension for e in ergebnisse]

session.close()

# Multiplot (2x2 Grid) erstellen
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Porenanalyse - ORM Multiplot Dashboard", fontsize=16, fontweight='bold')

# Plot 1: Porosität pro Bild
axes[0, 0].barh(noms, porositaeten, color='teal')
axes[0, 0].set_title("Porosität (Anteil)")
axes[0, 0].set_xlabel("Wert")

# Plot 2: Porenanzahl
axes[0, 1].bar(noms, poren_anzahlen, color='coral')
axes[0, 1].set_title("Porenanzahl")
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Mittlerer Porendurchmesser
axes[1, 0].plot(noms, durchmesser, marker='o', color='purple', linestyle='-')
axes[1, 0].set_title("Mittlerer Porendurchmesser (px)")
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Fraktale Dimension
axes[1, 1].scatter(porositaeten, fraktal_dims, color='navy', s=100)
axes[1, 1].set_title("Fraktale Dimension vs. Porosität")
axes[1, 1].set_xlabel("Porosität")
axes[1, 1].set_ylabel("Fraktale Dimension")

plt.tight_layout()
plt.show()
