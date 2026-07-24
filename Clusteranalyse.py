import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. DATEN LADEN (Aus dem Labor-Verzeichnis eures Containers)
# Wir holen die Messwerte für die Kokos-Aktivkohle und die Poren des Aerogels
particle_df = pd.read_excel(r'polymer analysis\particle sizes.xlsx')
pore_df = pd.read_excel(r'polymer analysis\pore size.xlsx')

# 2. DATEN IN FORM BRINGEN (Die Konzentration dient als Bindeglied)
# Wir formen die Tabellen um, damit wir sie sauber miteinander verbinden können
particle_long = particle_df.melt(var_name='Konzentration', value_name='Partikel_Groesse')
pore_long = pore_df.melt(var_name='Konzentration', value_name='Poren_Groesse')

# Konzentrationen als Fließkommazahl (Float) angleichen für das Matching
particle_long['Konzentration'] = particle_long['Konzentration'].astype(float)
pore_long['Konzentration'] = pore_long['Konzentration'].astype(float)

# Hier fügen wir beide Tabellen zusammen. 
# Jeder Datenpunkt hat jetzt: [Konzentration, Partikelgröße, Porengröße]
combined_df = pd.merge(particle_long, pore_long, on='Konzentration', how='inner')

# 3. FEATURES FÜR DEN K-MEANS-ALGORITHMUS DEFINIEREN
features = combined_df[['Konzentration', 'Partikel_Groesse', 'Poren_Groesse']]

# Standardisierung: Wir bringen alle Werte auf dieselbe Skala, 
# damit K-Means die Dimensionen fair vergleicht.
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. OPTIMALE ANZAHL AN CLUSTERN (K) BERECHNEN
# Weil der Datensatz klein ist, nutzen wir den Silhouetten-Koeffizienten
range_n_clusters = range(2, 6) # Mehr als 5 Materialzustände machen physikalisch keinen Sinn
silhouette_scores = []

for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, cluster_labels)
    silhouette_scores.append(score)

# Das mathematische Optimum für die Anzahl der Zustände (z.B. Opak, Transluzent, Transparent)
optimal_k = range_n_clusters[np.argmax(silhouette_scores)]
print(f"Optimaler Material-Zustand für die Matrix gefunden bei K = {optimal_k}")

# Endgültige Cluster-Zuteilung berechnen
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
combined_df['Cluster'] = kmeans_final.fit_predict(scaled_features)

# 5. DREIDIMENSIONALE VISUALISIERUNG (Der physikalische Zustand im Raum)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    combined_df['Konzentration'],
    combined_df['Partikel_Groesse'],
    combined_df['Poren_Groesse'],
    c=combined_df['Cluster'], cmap='viridis', s=60, alpha=0.8
)

ax.set_title("Strukturelle Cluster des Kokos-Agar-Acrylat-Gels", fontsize=14)
ax.set_xlabel('Alkohol-Konzentration (Methanol/IPA in %)')
ax.set_ylabel('Aktivkohle-Partikelgröße (nm)')
ax.set_zlabel('Aerogel-Porengröße (nm)')
plt.colorbar(scatter, label='Material-Zustand (Cluster-ID)')
plt.tight_layout()

# Grafik als PNG für die Dokumentation im Container speichern
plt.savefig('Clustering_Material_States.png', dpi=300)
plt.show()

# 6. DATEN-EXPORT (Saubere Speicherung der Ergebnisse)
# Statistische Kennwerte (Mittelwert, Standardabweichung) pro Cluster berechnen
cluster_stats = combined_df.groupby('Cluster').agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
cluster_stats.to_csv('cluster_stats_advanced.csv', index=False)

# Die echten Datenpunkte der einzelnen Cluster in separate Dateien schreiben
# Das G13X-Board zieht sich später diese Werte für die Laser-Kalibrierung
for i in range(optimal_k):
    echte_cluster_daten = combined_df[combined_df['Cluster'] == i]
    echte_cluster_daten.to_csv(f"cluster_{i}_messwerte.csv", index=False)

# 7. EXPERTEN-ANOMALIEN-ERKENNUNG (Ausreißer-Filter über IQR)
# Hier checken wir, ob beim Ablöschen mit der Asche Fehler im Gefüge entstanden sind
Q1 = combined_df['Partikel_Groesse'].quantile(0.25)
Q3 = combined_df['Partikel_Groesse'].quantile(0.75)
IQR = Q3 - Q1
untere_grenze = Q1 - 1.5 * IQR
obere_grenze = Q3 + 1.5 * IQR

# Wenn ein Punkt außerhalb der Grenzen liegt, wird er als Anomalie (1) markiert
combined_df['Anomalie'] = np.where(
    (combined_df['Partikel_Groesse'] < untere_grenze) | (combined_df['Partikel_Groesse'] > obere_grenze), 
    1, 0
)
print(f"Strukturfehler beim Asche-Ablöschen erkannt: {combined_df['Anomalie'].sum()} Instanzen.")
