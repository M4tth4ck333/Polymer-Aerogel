import os
import math
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage import io, measure, morphology, util
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
from skimage.segmentation import watershed

# Bildpfade (SEM-Aufnahmen)
image_files = [
    r"Aerogel_SEM\20230822_12p_21.tif",
    r"Aerogel_SEM\20230823_10p_17.tif",
    r"Aerogel_SEM\20230823_13p_17.tif",
    r"Aerogel_SEM\20230824_14p_18.tif",
    r"Aerogel_SEM\20230824_15p_18.tif",
    r"Aerogel_SEM\20230831_11p_17.tif"
]

# Graustufen-Einlesen & Normalisierung auf 8-Bit
def imread_gray_8bit(path: str):
    img = io.imread(path)
    if img.ndim == 3:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

# Bildverbesserung und -verarbeitung
def enhance_contrast(img): 
    return cv2.createCLAHE(2.0, (8, 8)).apply(img)

def denoise(img): 
    return cv2.bilateralFilter(img, 5, 10, 10)

def binarize_pores(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ad = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5)
    comb = ((otsu > 0).astype(np.uint8) + (ad > 0).astype(np.uint8)) >= 1
    return comb if 0.2 <= comb.mean() <= 0.95 else (~comb)

def clean_binary(bin_img): 
    # Expliziter Cast auf Bool für Kompatibilität mit neueren skimage-Versionen
    bool_img = bin_img.astype(bool)
    cleaned_obj = remove_small_objects(bool_img, max_size=25)
    cleaned_holes = remove_small_holes(cleaned_obj, area_threshold=25)
    return cleaned_holes

# Merkmalsanalyse
def compute_shape_descriptors(regions):
    circularities, aspect_ratios, solidities, eq_diams = [], [], [], []
    for r in regions:
        area = r.area
        perimeter = r.perimeter if r.perimeter > 0 else 1.0
        circularity = 4.0 * math.pi * area / (perimeter ** 2)
        minr, minc, maxr, maxc = r.bbox
        h = maxr - minr
        w = maxc - minc
        ar = max(h, w) / (min(h, w) + 1e-6)
        
        circularities.append(circularity)
        aspect_ratios.append(ar)
        solidities.append(r.solidity if hasattr(r, "solidity") else np.nan)
        # In neueren skimage Versionen: equivalent_diameter_area
        eq_diams.append(r.equivalent_diameter_area if hasattr(r, "equivalent_diameter_area") else r.equivalent_diameter)
        
    return (np.nanmean(circularities) if circularities else np.nan,
            np.nanmean(aspect_ratios) if aspect_ratios else np.nan,
            np.nanmean(solidities) if solidities else np.nan,
            eq_diams)

def distance_watershed(binary_mask):
    distance = ndi.distance_transform_edt(binary_mask)
    local_max = morphology.local_maxima(distance)
    markers, _ = ndi.label(local_max)
    return watershed(-distance, markers, mask=binary_mask)

def skeleton_metrics(solid_mask):
    skel = skeletonize(solid_mask.astype(bool))
    length_px = skel.sum()
    area_px = skel.size
    density = length_px / area_px if area_px > 0 else 0.0
    
    # Sicherer Rand über SciPy convolve mit Constant Mode (Pad mit 0)
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.int32)
    neigh = ndi.convolve(skel.astype(np.int32), kernel, mode='constant', cval=0)
    neighbors = neigh - 10
    
    branch_points = int(((neighbors >= 3) & skel).sum())
    end_points = int(((neighbors == 1) & skel).sum())
    return (density, branch_points, end_points)

def fractal_dimension(binary_mask):
    # Korrigiertes, verlässliches Box-Counting
    img = binary_mask > 0
    h, w = img.shape
    sizes, counts = [], []
    
    s = 1
    max_s = min(h, w)
    while s <= max_s // 4:
        nH = h // s
        nW = w // s
        cropped = img[:nH * s, :nW * s]
        
        # 4D-Reshape für echtes s x s Block-Summieren
        blocks = cropped.reshape(nH, s, nW, s)
        block_sum = blocks.any(axis=(1, 3))
        count = np.count_nonzero(block_sum)
        
        if count > 0:
            sizes.append(1.0 / s)
            counts.append(count)
        s *= 2
        
    if len(sizes) < 2:
        return np.nan
    return float(np.polyfit(np.log(sizes), np.log(counts), 1)[0])

# --- HAUPTVERARBEITUNGSSCHLEIFE ---
rows = []
for path in image_files:
    if not os.path.exists(path):
        print(f"Datei nicht gefunden: {path}")
        continue
        
    row = {"image_name": os.path.basename(path)}
    try:
        # 1. Einlesen & Vorverarbeitung
        raw = imread_gray_8bit(path)
        enhanced = enhance_contrast(raw)
        denoised = denoise(enhanced)
        
        # 2. Binarisierung (Poren = True)
        bin_pores = binarize_pores(denoised)
        clean_pores = clean_binary(bin_pores)
        solid_phase = ~clean_pores
        
        # 3. Porosität
        porosity = clean_pores.mean()
        row["porosity"] = porosity
        
        # 4. Watershed-Segmentierung
        labeled_pores = distance_watershed(clean_pores)
        regions = measure.regionprops(labeled_pores)
        
        # 5. Form-Deskriptoren & Geometrie
        mean_circ, mean_ar, mean_solidity, eq_diams = compute_shape_descriptors(regions)
        row["mean_circularity"] = mean_circ
        row["mean_aspect_ratio"] = mean_ar
        row["mean_solidity"] = mean_solidity
        row["pore_count"] = len(regions)
        row["mean_pore_diameter_px"] = np.nanmean(eq_diams) if eq_diams else np.nan
        row["median_pore_diameter_px"] = np.nanmedian(eq_diams) if eq_diams else np.nan
        
        # 6. Festkörper-Skelettierung
        skel_density, branches, endpoints = skeleton_metrics(solid_phase)
        row["skeleton_density"] = skel_density
        row["branch_points"] = branches
        row["end_points"] = endpoints
        
        # 7. Fraktale Dimension
        row["fractal_dimension"] = fractal_dimension(clean_pores)
        
        rows.append(row)
        print(f"Erfolgreich verarbeitet: {row['image_name']} | Porosität: {porosity*100:.2f}%")
        
    except Exception as e:
        print(f"Fehler bei {path}: {str(e)}")

# Ergebnisse in DataFrame speichern & exportieren
df_results = pd.DataFrame(rows)
df_results.to_csv("SEM_Aerogel_Morphology_Results.csv", index=False)
print("\nAnalyse abgeschlossen! Daten in 'SEM_Aerogel_Morphology_Results.csv' gespeichert.")
