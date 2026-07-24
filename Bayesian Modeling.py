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

warnings.filterwarnings('ignore')

# ========= SETTINGS =========
sns.set(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['figure.figsize'] = (12, 8)

# ========= DATENVALIDIERUNG =========
def validate_data(particle_df, pore_df, pva_df):
    try:
        particle_cols = ['concentration', 'particle_mean', 'particle_std']
        pore_cols = ['concentration', 'pore_mean', 'pore_std']
        pva_cols = ['concentration', 'density', 'volume_shrinkage', 'modulus', 'modulus_std', 'transmittance']
        
        if not all(col in particle_df.columns for col in particle_cols):
            raise ValueError("particle_df fehlen notwendige Spalten")
        if not all(col in pore_df.columns for col in pore_cols):
            raise ValueError("pore_df fehlen notwendige Spalten")
        if not all(col in pva_df.columns for col in pva_cols):
            raise ValueError("pva_df fehlen notwendige Spalten")
        
        for df, name in [(particle_df, 'particle_df'), (pore_df, 'pore_df'), (pva_df, 'pva_df')]:
            if df.isnull().any().any():
                raise ValueError(f"{name} enthält fehlende Werte (NaN)")
            if not (df['concentration'] > 0).all():
                raise ValueError(f"{name}: Konzentrationen müssen positiv sein")
        
        print("[INFO] Datenvalidierung erfolgreich abgeschlossen.")
        return True
    except Exception as e:
        print(f"[ERROR] Datenvalidierung fehlgeschlagen: {e}")
        return False

# ========= DATENANREICHERUNG (DATA AUGMENTATION) =========
def expand_concentration_range(data, target_samples=200, noise_level=0.02):
    current_samples = len(data)
    expansion_factor = max(1, (target_samples - current_samples) // current_samples)
    new_rows = []
    
    for _, row in data.iterrows():
        for _ in range(expansion_factor):
            new_row = row.copy()
            new_row['concentration'] += np.random.normal(0, 0.005)
            for col in ['pore_mean', 'particle_mean', 'density', 'volume_shrinkage']:
                new_row[col] += np.random.normal(0, noise_level * row[col])
            
            # Abgeleitete Merkmale neu berechnen
            new_row['pore_particle_ratio'] = new_row['pore_mean'] / (new_row['particle_mean'] + 1e-8)
            new_row['pore_particle_diff'] = new_row['pore_mean'] - new_row['particle_mean']
            
            for target in ['modulus', 'transmittance']:
                new_row[target] += np.random.normal(0, noise_level * row[target])
            new_rows.append(new_row)
            
    expanded_df = pd.DataFrame(new_rows)
    return pd.concat([data, expanded_df], ignore_index=True)

# ========= DATEN LADEN =========
def load_and_preprocess_data(expand=True, target_samples=200):
    try:
        particle_df = pd.read_excel("particle sizes.xlsx", sheet_name="Sheet2").iloc[1:, :]
        particle_df.columns = ['concentration'] + [f'particle_{i}' for i in range(1, 41)] + ['particle_mean', 'particle_std']

        pore_df = pd.read_excel("pore size.xlsx", sheet_name="Sheet4").iloc[1:, :]
        pore_df.columns = ['concentration'] + [f'pore_{i}' for i in range(1, 26)] + ['pore_mean', 'pore_std']

        pva_df = pd.read_excel("PVA Aerogel Data.xlsx", sheet_name="Sheet2")
        pva_df.columns = ['concentration', 'density', 'volume_shrinkage', 'modulus', 'modulus_std', 'transmittance']

        if not validate_data(particle_df, pore_df, pva_df):
            return None

        merged_df = pd.merge(particle_df, pore_df, on='concentration', how='inner')
        final_df = pd.merge(merged_df, pva_df, on='concentration', how='inner')
        final_df['pore_particle_ratio'] = final_df['pore_mean'] / final_df['particle_mean']
        final_df['pore_particle_diff'] = final_df['pore_mean'] - final_df['particle_mean']

        if expand:
            print(f"[INFO] Ursprüngliche Stichprobengröße: {len(final_df)}")
            final_df = expand_concentration_range(final_df, target_samples)
            print(f"[INFO] Erweiterte Stichprobengröße: {len(final_df)}")

        return final_df
    except Exception as e:
        print(f"[ERROR] Fehler beim Laden der Daten: {e}")
        return None

# ========= BAYES-MODELL (MCMC-STABILISIERT) =========
def build_hierarchical_model(data):
    if data is None:
        return None, None

    conc = data['concentration'].values
    density = data['density'].values
    shrinkage = data['volume_shrinkage'].values
    pore_mean = data['pore_mean'].values
    particle_mean = data['particle_mean'].values
    modulus = data['modulus'].values
    transmittance = data['transmittance'].values

    with pm.Model() as model:
        # Priors für Konzentrationseinfluss
        beta_conc_pore = pm.Normal('beta_conc_pore', mu=0, sigma=2)
        beta_conc_part = pm.Normal('beta_conc_part', mu=0, sigma=2)
        
        # Gezielte Modellierung der Differenz statt starrer Penalty
        diff_pore_part = pm.HalfNormal('diff_pore_part', sigma=10)
        
        # Regression für E-Modul
        alpha_E = pm.Normal('alpha_E', mu=0, sigma=10)
        beta_E = pm.Normal('beta_E', mu=0, sigma=2, shape=4)
        mu_E = alpha_E + beta_E[0]*density + beta_E[1]*shrinkage + beta_E[2]*pore_mean + beta_E[3]*particle_mean
        sigma_E = pm.HalfNormal('sigma_E', sigma=5)
        pm.Normal('E_obs', mu=mu_E, sigma=sigma_E, observed=modulus)

        # Regression für Transparenz
        alpha_T = pm.Normal('alpha_T', mu=0, sigma=10)
        beta_T = pm.Normal('beta_T', mu=0, sigma=2, shape=4)
        mu_T = alpha_T + beta_T[0]*density + beta_T[1]*shrinkage + beta_T[2]*pore_mean + beta_T[3]*particle_mean
        sigma_T = pm.HalfNormal('sigma_T', sigma=5)
        pm.Normal('T_obs', mu=mu_T, sigma=sigma_T, observed=transmittance)

        try:
            trace = pm.sample(1000, tune=1000, target_accept=0.95, chains=2, cores=2, return_inferencedata=True)
        except Exception as e:
            print(f"[ERROR] Bayes-Sampling fehlgeschlagen: {e}")
            return None, None

    return trace, model

# ========= RANDOM FOREST MODELLIERUNG =========
def build_ml_models(data, features):
    if data is None:
        return None, None

    X = data[features] # Direkt unskaliert für saubere Baum-Interpretierbarkeit
    y_modulus = data['modulus'].values
    y_trans = data['transmittance'].values

    try:
        rf_modulus = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_modulus)
        rf_trans = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_trans)

        print(f"[RF] E-Modul R² Score: {r2_score(y_modulus, rf_modulus.predict(X)):.4f}")
        print(f"[RF] Transmission R² Score: {r2_score(y_trans, rf_trans.predict(X)):.4f}")

        return rf_modulus, rf_trans
    except Exception as e:
        print(f"[ERROR] Random Forest Training fehlgeschlagen: {e}")
        return None, None

# ========= SHAP ANALYSE =========
def shap_analysis(model, X, feature_names, title):
    if model is None:
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    plt.title(f"SHAP Impact Analysis - {title}")
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"results/shap_summary_{title}.png", dpi=300)
    plt.close()
    print(f"[INFO] SHAP-Analyse gespeichert: {title}")

# ========= HAUPTPROZESS =========
def main():
    os.makedirs("results", exist_ok=True)
    features = ['concentration', 'density', 'volume_shrinkage', 'pore_mean', 'particle_mean', 'pore_particle_ratio']
    
    data = load_and_preprocess_data(expand=True, target_samples=200)
    if data is not None:
        trace, bayes_model = build_hierarchical_model(data)
        rf_mod, rf_trans = build_ml_models(data, features)
        
        if trace is not None:
            az.plot_trace(trace, var_names=['alpha_E', 'beta_E', 'alpha_T', 'beta_T'])
            plt.tight_layout()
            plt.savefig("results/bayes_trace_plot.png", dpi=300)
            plt.close()
            az.summary(trace).to_csv("results/bayes_summary.csv")
            
        if rf_mod and rf_trans:
            X = data[features]
            shap_analysis(rf_mod, X, features, "Modulus")
            shap_analysis(rf_trans, X, features, "Transmittance")
            
        data.to_csv("results/enhanced_dataset.csv", index=False)
        print("[SUCCESS] Gesamte Daten- & Modellpipeline erfolgreich ausgeführt!")

if __name__ == "__main__":
    main()
