import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import shap

# SQLAlchemy für das erweiterte kartesische ORM
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

warnings.filterwarnings('ignore')

# ========= SETTINGS =========
sns.set(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['figure.figsize'] = (12, 8)

# ========= 1. ERWEITERTES KARTESISCHES ORM (DATENBANK-MODELL) =========
Base = declarative_base()

class ProbeEintrag(Base):
    __tablename__ = 'proben_meta'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    bild_name = Column(String, unique=True)
    porositaet_px = Column(Float)
    poren_anzahl = Column(Integer)
    mittlerer_poren_durchmesser_px = Column(Float)
    skelett_laengen_dichte = Column(Float)
    fraktale_dimension = Column(Float)
    
    # Beziehung zu kartesischen Gitterzellen
    gitter_zellen = relationship("KartesischeGitterZelle", back_populates="probe", cascade="all, delete-orphan")

class KartesischeGitterZelle(Base):
    __tablename__ = 'kartesisches_gitter'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    probe_id = Column(Integer, ForeignKey('proben_meta.id'))
    
    # Kartesische Koordinaten im 3D-Volumen (x, y, z)
    x_pos = Column(Float)
    y_pos = Column(Float)
    z_pos = Column(Float)
    
    # Lokale Navier-Stokes & Fluid-Parameter
    lokale_porositaet = Column(Float)
    lokal_permeabilitaet = Column(Float)  # m² (aus Navier-Stokes / Darcy-Gesetz)
    druckgradient = Column(Float)          # Pa/m
    reynolds_zahl = Column(Float)          # Re (Strömungsregime)
    
    probe = relationship("ProbeEintrag", back_populates="gitter_zellen")

# SQLite-Datenbank mit kartesischer Raumstruktur initialisieren
engine = create_engine('sqlite:///aerogel_kartesisch_navierstokes.db', echo=False)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ========= 2. NAVIER-STOKES & POREN-STRÖMUNGS-MODELLIERUNG =========
def berechne_navier_stokes_stroemung(porositaet, mittlerer_poren_durchmesser, druckabfall=1000.0, dynamische_viskositaet=0.001):
    """
    Modelliert die stationäre, viskose Strömung (vereinfachte Navier-Stokes / Hagen-Poiseuille / Darcy-Ansätze)
    durch das mikroskopische Porennetzwerk des Aerogels.
    """
    # Effektiver hydraulischer Radius (r = D_poren / 2)
    r_hyd = (mittlerer_poren_durchmesser * 1e-6) / 2.0  # Annahme: Skalierung in Mikrometern
    
    # Permeabilität nach Carman-Kozeny-Modell gekoppelt an Porosität
    # k = (phi^3 * r^2) / (C * (1 - phi)^2)
    c_kozeny = 5.0
    if porositaet >= 1.0: porositaet = 0.99
    if porositaet <= 0.0: porositaet = 0.01
    
    permeabilitaet = (porositaet**3 * r_hyd**2) / (c_kozeny * (1.0 - porositaet)**2 + 1e-12)
    
    # Mittlere Strömungsgeschwindigkeit v = (k / mu) * (Delta_p / L)
    # Annahme Standard-Länge L = 1e-3 m (1 mm Probe)
    l_probe = 1e-3
    geschwindigkeit = (permeabilitaet / dynamische_viskositaet) * (druckabfall / l_probe)
    
    # Lokale Reynolds-Zahl (Re = rho * v * d / mu), Dichte von Luft/Fluid ~ 1.2 kg/m³
    rho_fluid = 1.2
    reynolds_zahl = (rho_fluid * geschwindigkeit * (2 * r_hyd)) / dynamische_viskositaet
    
    return float(permeabilitaet), float(druckabfall / l_probe), float(reynolds_zahl)


# ========= 3. DATENANREICHERUNG & SIMULATION =========
defexpand_and_simulate(data):
    """
    Reichert die Daten an und generiert ein synthetisches kartesisches 3D-Gitter 
    mit Navier-Stokes-Strömungsparametern für jede Probe.
    """
    session = Session()
    
    for _, row in data.iterrows():
        bild_name = str(row['concentration']) # Platzhalter für Bildnamen
        
        # In DB speichern oder holen
        probe = session.query(ProbeEintrag).filter_by(bild_name=bild_name).first()
        if not probe:
            probe = ProbeEintrag(
                bild_name=bild_name,
                porositaet_px=float(row['pore_particle_ratio']),
                poren_anzahl=int(row.get('pore_mean', 50)),
                mittlerer_poren_durchmesser_px=float(row['pore_mean']),
                skelett_laengen_dichte=float(row.get('density', 0.5)),
                fraktale_dimension=2.3
            )
            session.add(probe)
            session.commit()
        
        # Kartesisches Gitter im 3D-Raum (z.B. 3x3x3 = 27 Gitterpunkte pro Probe)
        # Um die Navier-Stokes-Verteilung räumlich abzubilden
        if not probe.gitter_zellen:
            for x in np.linspace(0, 1, 3):
                for y in np.linspace(0, 1, 3):
                    for z in np.linspace(0, 1, 3):
                        # Lokale Variation einbauen
                        lokale_porositaet = np.clip(probe.porositaet_px + np.random.normal(0, 0.05), 0.1, 0.9)
                        perm, dp_dx, re = berechne_navier_stokes_stroemung(
                            lokale_porositaet, 
                            probe.mittlerer_poren_durchmesser_px
                        )
                        
                        gitter_zelle = KartesischeGitterZelle(
                            probe_id=probe.id,
                            x_pos=float(x),
                            y_pos=float(y),
                            z_pos=float(z),
                            lokale_porositaet=lokale_porositaet,
                            lokal_permeabilitaet=perm,
                            druckgradient=dp_dx,
                            reynolds_zahl=re
                        )
                        session.add(gitter_zelle)
            session.commit()
            
    session.close()
    print("[INFO] Kartesisches Navier-Stokes-Gitter erfolgreich in SQLite-DB geschrieben.")


# ========= HAUPTPROZESS MIT ORM-AUSLESUNG =========
def main():
    os.makedirs("results", exist_ok=True)
    
    # Simulieren wir einen DataFrame für den Testlauf
    np.random.seed(42)
    n = 10
    mock_data = pd.DataFrame({
        'concentration': np.linspace(0.05, 0.5, n),
        'pore_mean': np.random.uniform(5, 20, n),
        'particle_mean': np.random.uniform(2, 10, n),
        'density': np.random.uniform(0.1, 0.4, n),
        'volume_shrinkage': np.random.uniform(0.05, 0.2, n),
        'modulus': np.random.uniform(10, 100, n),
        'transmittance': np.random.uniform(50, 95, n)
    })
    mock_data['pore_particle_ratio'] = mock_data['pore_mean'] / mock_data['particle_mean']
    
    # Ausführung der kartesischen Navier-Stokes-Modellierung
    expand_and_simulate(mock_data)
    
    # Abfrage aus dem kartesischen ORM zur Verifizierung
    session = Session()
    ergebnis_zellen = session.query(KartesischeGitterZelle).limit(5).all()
    print("\n[DB-CHECK] Beispielhafte Gitterzellen aus der SQLite-Datenbank:")
    for z in ergebnis_zellen:
        print(f"Koordinate: ({z.x_pos}, {z.y_pos}, {z.z_pos}) | Permeabilität: {z.lokal_permeabilitaet:.2e} m² | Re: {z.reynolds_zahl:.5f}")
    session.close()

if __name__ == "__main__":
    main()
