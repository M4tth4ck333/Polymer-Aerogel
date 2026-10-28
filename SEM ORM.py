import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, measure, morphology, util
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from scipy import ndimage as ndi

# SQLAlchemy für das erweiterte kartesische ORM
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# --- 1. ERWEITERTES KARTESISCHES ORM (DATENBANK-MODELL) ---
Base = declarative_base()

class AerogelMatrixEintrag(Base):
    __tablename__ = 'aerogel_matrix_proben'
    
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
    
    # Beziehung zum räumlichen Navier-Stokes-Gitter im 3D-Volumen
    fluid_gitter = relationship("KartesischesFluidGitter", back_populates="matrix_probe", cascade="all, delete-orphan")

class KartesischesFluidGitter(Base):
    __tablename__ = 'kartesisches_fluid_gitter'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    probe_id = Column(Integer, ForeignKey('aerogel_matrix_proben.id'))
    
    # Kartesische Koordinaten im 3D-Volumen (x, y, z)
    x_pos = Column(Float)
    y_pos = Column(Float)
    z_pos = Column(Float)
    
    # Lokale Navier-Stokes & Fluid-Parameter
    lokale_porositaet = Column(Float)
    lokal_permeabilitaet = Column(Float)  # m² (aus Navier-Stokes / Carman-Kozeny)
    druckgradient = Column(Float)          # Pa/m
    reynolds_zahl = Column(Float)          # Re (Strömungsregime)
    
    matrix_probe = relationship("AerogelMatrixEintrag", back_populates="fluid_gitter")

# SQLite-Datenbank mit kartesischer Raumstruktur initialisieren
engine = create_engine('sqlite:///aerogel_kartesisch_navierstokes.db', echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# --- 2. BILDVERARBEITUNGS- & NAVIER-STOKES-FUNKTIONEN ---
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

def berechne_navier_stokes_stroemung(porositaet, mittlerer_poren_durchmesser, druckabfall=1000.0, dynamische_viskositaet=0.001):
    """
    Modelliert die stationäre, viskose Strömung (Carman-Kozeny / Navier-Stokes-Näherung)
    durch das mikroskopische Porennetzwerk basierend auf den SEM-Metriken.
    """
    r_hyd = (mittlerer_poren_durchmesser * 1e-6) / 2.0  # Skalierung in Mikrometern
    c_kozeny = 5.0
    if porositaet >= 1.0: porositaet = 0.99
    if porositaet <= 0.0: porositaet = 0.01
    
    permeabilitaet = (porositaet**3 * r_hyd**2) / (c_kozeny * (1.0 - porositaet)**2 + 1e-12)
    l_probe = 1e-3
    geschwindigkeit = (permeabilitaet / dynamische_viskositaet) * (druckabfall / l_probe)
