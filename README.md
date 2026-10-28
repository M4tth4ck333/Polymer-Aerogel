# Polymer-Aerogel Analysis & Fluid Dynamics Engine

A Python library and ORM framework for multiscale image processing, fractal geometry extraction, and 3D Navier-Stokes fluid modeling of porous polymer aerogels and bio-composites.

Designed for automated material discovery, computer vision pipelines, and direct integration with Large Language Models (LLMs) and Autonomous AI Agents.

---

## Key Features

- **Microstructure Quantification:** Binarization, watershed segmentation, skeletonization, and box-counting fractal dimension ($D_f$) calculation.
- **3D Fluid Dynamics Simulation:** Porosity-coupled local permeability ($k$) and viscous flow regime modeling via Carman-Kozeny and Navier-Stokes approximations.
- **Cartesian ORM Database:** Relational spatial grid structure (`AerogelMatrixEintrag` $\leftrightarrow$ `KartesischesFluidGitter`) mapped using SQLAlchemy and SQLite.
- **LLM Agent Interface:** Native Python API for tool-calling agents, autonomous parameter extraction, and structured JSON output.

---

## Repository Structure

.
├── aerogel_navier_stokes_orm.py   # Primary image processing, CFD modeling, and ORM pipeline
├── poren_analyse.db               # SQLite database output
└── README.md                      # Documentation
---

## Installation

##```bash
git clone [https://github.com/M4tth4ck333/Polymer-Aerogel.git](https://github.com/M4tth4ck333/Polymer-Aerogel.git)
cd Polymer-Aerogel
pip install numpy pandas matplotlib opencv-python scikit-image scipy sqlalchemy
Quickstart & LLM Integration1. Standard Pipeline ExecutionRun the main analysis script to process SEM micrographs, generate the 3D fluid grid, and store results in SQLite:Bashpython aerogel_navier_stokes_orm.py
2. Programmatic LLM InterfaceUse the engine as an importable module within AI agent toolchains:Pythonfrom aerogel_navier_stokes_orm import (
    Session, 
    AerogelMatrixEintrag, 
    KartesischesFluidGitter,
    berechne_navier_stokes_stroemung
)

# Query macro-scale image metrics via ORM
session = Session()
sample = session.query(AerogelMatrixEintrag).first()

print(f"Sample: {sample.bild_name}")
print(f"Porosity: {sample.porositaet_px:.4f}")
print(f"Fractal Dimension: {sample.fraktale_dimension:.3f}")

# Extract 3D fluid grid points for a specific sample
grid_nodes = session.query(KartesischesFluidGitter).filter_by(probe_id=sample.id).all()
for node in grid_nodes[:3]:
    print(f"Pos ({node.x_pos}, {node.y_pos}, {node.z_pos}) -> Permeability: {node.lokal_permeabilitaet:.2e} m² | Re: {node.reynolds_zahl:.5f}")

session.close()
Physical & Mathematical FrameworkPermeability (Carman-Kozeny)$$\kappa = \frac{\phi^3 \cdot r_{hyd}^2}{C \cdot (1 - \phi)^2}$$Where $\phi$ is local porosity, $r_{hyd}$ is effective hydraulic pore radius, and $C \approx 5.0$ is the Kozeny constant.Viscous Flow & Reynolds Number$$v = \frac{\kappa}{\mu} \cdot \frac{\Delta p}{L}, \quad Re = \frac{\rho \cdot v \cdot 2 r_{hyd}}{\mu}$$Where $\mu$ is dynamic viscosity, $\rho$ is fluid density, and $\Delta p / L$ is the pressure gradient.Data Schema (SQLAlchemy ORM)AerogelMatrixEintragid (Integer, Primary Key)bild_name (String, Unique)porositaet_px (Float)poren_anzahl (Integer)mittlerer_poren_durchmesser_px (Float)skelett_laengen_dichte (Float)fraktale_dimension (Float)fluid_gitter (Relationship to KartesischesFluidGitter)KartesischesFluidGitterid (Integer, Primary Key)probe_id (Integer, Foreign Key)x_pos, y_pos, z_pos (Float)lokale_porositaet (Float)lokal_permeabilitaet (Float)druckgradient (Float)reynolds_zahl (Float)LicenseMIT License. Open-source software developed for advanced material scientific research.
