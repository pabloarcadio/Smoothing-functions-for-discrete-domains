# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:44:44 2026

@author: user
"""
#Prologue:
#You can find here all the required functions to run 
#the smoothing filters over a set of images and its evaluation
#run this file once before runnning "suavizados multiples.py" or
#"suavizados multiples rgb.py"

#------------------------------
#%% 0. PRE-PROCESSING FUNCTIONS
#------------------------------
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import time

def sal_y_pimienta(imagen, cantidad=0.05):
    salida = np.copy(imagen)
    num_salt = np.ceil(cantidad * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in imagen.shape]
    salida[tuple(coords)] = 255
    num_pepper = np.ceil(cantidad * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in imagen.shape]
    salida[tuple(coords)] = 0
    return salida

def sobel_universal(image, threshold, ksize=3):
    if len(image.shape) == 3:  # Caso RGB (H, W, 3)
        h, w, c = image.shape
        mags = []
        for i in range(c):
            ch = image[:, :, i]
            gx = cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=ksize)
            gy = cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=ksize)
            mags.append(np.sqrt(gx**2 + gy**2))
        mag_final = np.max(np.array(mags), axis=0)
    else:  # Caso Grises (H, W)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
        mag_final = np.sqrt(gx**2 + gy**2)

    _, binarizada = cv2.threshold(np.uint8(np.clip(mag_final, 0, 255)), threshold, 255, cv2.THRESH_BINARY)
    return binarizada

# ----------------------
# 1. GRAY SMOOTHING FILTERS
# ----------------------


# 1.1 Classic mean
def classic_mean(image, kernel_size):
    """
    Aplica un filtro de promedio 2D a una imagen en escala de grises con padding.

    Args:
        image: La imagen de entrada como un array NumPy.
        kernel_size: El tamaño del kernel (debe ser impar).

    Returns:
        La imagen filtrada como un array NumPy del mismo tamaño (gracias al padding) que la original.
    """

    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Calcular el margen del kernel
    kernel_margin = kernel_size // 2

    # Crear una imagen con padding (corrección aquí)
    padded_image = np.pad(image, ((kernel_margin, kernel_margin), (kernel_margin, kernel_margin)), mode='reflect')

    # Crear una imagen de salida con las mismas dimensiones que la original
    filtered_image = np.zeros_like(image)

    # Iterar sobre cada píxel de la imagen original
    for i in range(height):
        for j in range(width):
            # Extraer la ventana del kernel de la imagen con padding
            window = padded_image[i:i + kernel_size, j:j + kernel_size]

            # Calcular el promedio de los píxeles en la ventana
            gray_mean = np.mean(window) # Corrección aquí también

            # Asignar el promedio al píxel de salida
            filtered_image[i, j] = round(gray_mean) # Corrección aquí también

    return filtered_image

# 1.2 Discrete Mean Trunc
def discrete_mean_trunc(image, kernel_size, k_penalty=100):
    """
    Media Robusta por Optimización Discreta.
    Trunca la penalización de las diferencias que superan k_penalty.
    """
    if image.ndim != 2:
        raise ValueError("Imagen debe ser 2D (grises).")

    h, w = image.shape
    r = kernel_size // 2  # Radio del kernel
    padded = np.pad(image, ((r, r), (r, r)), mode='reflect')

    out = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            # Extraer ventana y aplanar
            window = padded[i:i+kernel_size, j:j+kernel_size]
            vals = window.ravel().astype(np.int32)

            amin = int(vals.min())
            amax = int(vals.max())

            best_a = amin
            best_err = np.iinfo(np.int64).max

            # Evaluar candidatos para el valor del píxel central
            for a in range(amin, amax + 1):
                # 1. Calculamos la diferencia absoluta
                diffs = np.abs(vals - a)
                
                # 2. Aplicamos el truncamiento (tu lógica del tope k)
                # Si la diferencia > k_penalty, se queda en k_penalty
                diffs_capped = np.minimum(diffs, k_penalty)
                
                # 3. Calculamos la suma de cuadrados de esas diferencias truncadas
                # Usamos int64 para evitar cualquier overflow en la suma
                total = np.sum(diffs_capped.astype(np.int64)**2)
                
                if total < best_err:
                    best_err = total
                    best_a = a

            out[i, j] = np.uint8(best_a)

    return out

# 1.3 Discrete Median truncated k penalization
def discrete_median_trunc(image, kernel_size, k_penalty):
    """
    Robust Discrete Median (exhaustive search)
    Minimiza la suma de diferencias absolutas TRUNCADAS a k_penalty.
    
    image: 2D uint8
    kernel_size: impar
    k_penalty: límite de penalización (threshold de robustez)
    """
    if image.ndim != 2:
        raise ValueError("Imagen debe ser 2D (grises).")

    h, w = image.shape
    r = kernel_size // 2  # Radio del kernel
    padded = np.pad(image, ((r, r), (r, r)), mode='reflect')

    out = np.zeros_like(image, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            vals = window.ravel().astype(np.int32)

            amin = int(vals.min())
            amax = int(vals.max())

            best_a = amin
            best_err = np.iinfo(np.int64).max

            # Búsqueda exhaustiva del candidato 'a'
            for a in range(amin, amax + 1):
                # 1. Diferencia absoluta (Norma L1)
                abs_diffs = np.abs(vals - a)
                
                # 2. Aplicamos el truncamiento (TU VERSIÓN EXTRA ROBUSTA)
                # Si la distancia es mayor a k, la penalización se estanca en k
                capped_diffs = np.minimum(abs_diffs, k_penalty)
                
                # 3. Sumamos las penalizaciones
                total = np.sum(capped_diffs.astype(np.int64))

                if total < best_err:
                    best_err = total
                    best_a = a

            out[i, j] = np.uint8(best_a)

    return out

# 1.4 Trimmed mean
def trimmed_mean(image, kernel_size, d=1):
    """
    Filtro de Media Alpha-Trimada.
    d: número de píxeles a eliminar en cada extremo (0 <= d < n/2)
    """
    if image.ndim != 2:
        raise ValueError("Imagen debe ser 2D.")
    
    h, w = image.shape
    r = kernel_size // 2
    padded = np.pad(image, ((r, r), (r, r)), mode='reflect')
    
    out = np.zeros_like(image, dtype=np.uint8)
    n = kernel_size * kernel_size

    if 2 * d >= n:
        raise ValueError("d es demasiado grande, no quedarían píxeles para promediar.")

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            # 1. Aplanar y ordenar los valores de menor a mayor
            sorted_vals = np.sort(window.ravel())
            
            # 2. Recortar d elementos de cada lado
            # Si d=1 y n=9, tomamos del índice 1 al 7 (8 excluido)
            trimmed_vals = sorted_vals[d : n - d]
            
            # 3. Calcular la media de los valores restantes
            out[i, j] = np.uint8(np.mean(trimmed_vals))
            
    return out

# ----------------------
# 2. RGB SMOOTHING FILTERS
# ----------------------

#2.1 discrete trunc decomposable (valid for L1 and L2)
from numba import jit

# 1. El motor matemático compilado en C (¡Por esto tarda 3 segundos!)
@jit(nopython=True)
def _solve_penalty_1d_k(vector, k_penalty, power=2):
    v_min, v_max = vector.min(), vector.max()
    best_val = v_min
    min_total_cost = 1e15
    
    # L2 si power=2, L1 si power=1
    penalty_cap = float(k_penalty)**power 
    
    for a in range(v_min, v_max + 1):
        total_cost = 0.0
        for x in vector:
            dist = np.abs(float(x) - float(a))**power
            total_cost += min(dist, penalty_cap)
            
        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_val = a
            
    return best_val

# 2. La función envoltorio (Wrapper)
def discrete_trunc_decomp(image, window_size, k_penalty, mode='mean'):
    # Selecciona la potencia según el modo (Mean = L2 = 2)
    p = 2 if mode == 'mean' else 1
    
    # Detecta si es Gris o RGB para no fallar
    is_gray = len(image.shape) == 2
    if is_gray:
        h, w = image.shape
        c = 1
        image = image[:, :, np.newaxis] # Expandir para que el bucle funcione
    else:
        h, w, c = image.shape
        
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    
    # Recorrido por la imagen
    for i in range(h):
        for j in range(w):
            for ch in range(c):
                # Extrae los 9 píxeles (si es 3x3)
                vec = padded[i:i+window_size, j:j+window_size, ch].flatten()
                # Llama a la función ultrarrápida compilada
                out[i, j, ch] = _solve_penalty_1d_k(vec, k_penalty, p)
                
    # Devuelve el formato original (Gris 2D o RGB 3D)
    return out[:, :, 0] if is_gray else out

# 2.2 Discrete Non decomposable (for L1 and L2)
@jit(nopython=True)
def _solve_penalty_3d_k(window_flat, k, power=2):
    r_min, r_max = window_flat[:,0].min(), window_flat[:,0].max()
    g_min, g_max = window_flat[:,1].min(), window_flat[:,1].max()
    b_min, b_max = window_flat[:,2].min(), window_flat[:,2].max()
    
    best_rgb = np.array([r_min, g_min, b_min], dtype=np.uint8)
    min_total_cost = 1e15
    penalty_cap = k**power
    
    for r in range(r_min, r_max + 1):
        for g in range(g_min, g_max + 1):
            for b in range(b_min, b_max + 1):
                total_cost = 0.0
                for idx in range(window_flat.shape[0]):
                    # Distancia vectorial (Euclídea)
                    dr = window_flat[idx, 0] - r
                    dg = window_flat[idx, 1] - g
                    db = window_flat[idx, 2] - b
                    
                    # Distancia según la potencia (L2^2 para Mean, L2 para Median)
                    if power == 2:
                        d_vec = (dr**2 + dg**2 + db**2)
                    else:
                        d_vec = np.sqrt(dr**2 + dg**2 + db**2)
                    
                    total_cost += min(d_vec, penalty_cap)
                
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_rgb[0], best_rgb[1], best_rgb[2] = r, g, b
    return best_rgb

def discrete_trunc_nondecomp(image, window_size, k, mode='mean'):
    p = 2 if mode == 'mean' else 1
    h, w, _ = image.shape
    pad = window_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size].reshape(-1, 3)
            out[i, j] = _solve_penalty_3d_k(window.astype(np.float32), k, p)
    return out

#2.3 Trimmed Mean RGB
@jit(nopython=True)
def _solve_trimmed(vec, d):
    # 1. Ordenar de menor a mayor
    vec_sorted = np.sort(vec)
    n = len(vec)
    # 2. Recortar d elementos de cada lado
    trimmed = vec_sorted[d : n - d]
    # 3. Calcular la media
    return np.mean(trimmed)

def trimmed_mean_rgb(image, kernel_size, d=1):
    is_gray = len(image.shape) == 2
    if is_gray:
        image = image[:, :, np.newaxis]
        
    h, w, c = image.shape
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    
    n = kernel_size * kernel_size
    if 2 * d >= n:
        raise ValueError("d es demasiado grande, no quedarían píxeles para promediar.")
        
    for i in range(h):
        for j in range(w):
            for ch in range(c):
                vec = padded[i:i+kernel_size, j:j+kernel_size, ch].flatten()
                out[i, j, ch] = np.uint8(np.round(_solve_trimmed(vec, d)))
                
    return out[:, :, 0] if is_gray else out

#2.4 Classic Median RGB decomposable
# Classic Median cada canal por separado)

def classic_median_rgb(image, window_size):
    return cv2.medianBlur(image, window_size)

#2.5a Vector Median Filter (VMF) - No decomponible
import numpy as np
from numba import jit

@jit(nopython=True)
def _vmf_logic(window):
    # n es el número de píxeles en la ventana (ej. 9 para un kernel de 3x3)
    n = window.shape[0]
    dists = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Cálculo de la distancia L2 (Euclidiana)
            diff = window[i] - window[j]
            dist_l2 = np.sqrt(np.sum(diff**2))
            dists[i] += dist_l2
            
    # Retorna el vector que minimiza la suma de distancias Euclidianas
    return window[np.argmin(dists)]


def vector_median_filter(image, window_size):
    h, w, c = image.shape
    pad = window_size // 2
    # Padding reflect para evitar artefactos en los bordes
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    
    for i in range(h):
        for j in range(w):
            # Extraemos la vecindad y la aplanamos a (N, Canales)
            window = padded[i:i+window_size, j:j+window_size].reshape(-1, c)
            out[i, j] = _vmf_logic(window.astype(np.float32))
            
    return out

#2.5b Vector Median Filter (VMF) L1 - No decomponible
@jit(nopython=True)
def _vmf_l1_logic(window):
    n = window.shape[0]
    dists = np.zeros(n)
    
    for i in range(n):
        for j in range(i + 1, n): # Simetría: evita recalcular distancias
            # Norma L1 (Manhattan)
            dist_l1 = np.sum(np.abs(window[i] - window[j]))
            
            dists[i] += dist_l1
            dists[j] += dist_l1
            
    return window[np.argmin(dists)]

def vector_median_filter_L1(image, window_size):
    h, w, c = image.shape
    pad = window_size // 2
    
    # 1. Padding reflect (Correcto, es lo mejor para bordes)
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    
    # 2. CONVERSIÓN PREVIA (Clave para la velocidad)
    # Convertimos toda la imagen a float32 aquí, fuera de los bucles
    padded_f32 = padded.astype(np.float32)
    
    out = np.zeros_like(image)
    
    # 3. Bucles de procesamiento
    for i in range(h):
        for j in range(w):
            # Extraemos la vecindad
            window = padded_f32[i:i+window_size, j:j+window_size].reshape(-1, c)
            
            # Llamamos a la lógica L1 optimizada
            out[i, j] = _vmf_l1_logic(window)
            
    return out

#2.5c Vector Median Filter Truncated version - No decomponible
import numpy as np
from numba import jit

@jit(nopython=True)
def _truncated_vmf_logic(window, k):
    # n es el número de píxeles en la ventana
    n = window.shape[0]
    dists = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # Cálculo de la distancia L2 (Euclidiana)
            diff = window[i] - window[j]
            dist_l2 = np.sqrt(np.sum(diff**2))
            
            # --- LÓGICA DE TRUNCACIÓN (Truncated) ---
            # Si la distancia supera K, se queda en K.
            # Esto ignora la magnitud del ruido si es "demasiado" ruidoso.
            dists[i] += min(dist_l2, k)
            
    # Retorna el vector que minimiza la suma de distancias truncadas
    return window[np.argmin(dists)]

def truncated_vector_median_filter(image, window_size, k=200.0):
    h, w, c = image.shape
    pad = window_size // 2
    # Padding reflect para evitar artefactos en los bordes
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    out = np.zeros_like(image)
    
    # Pre-convertir a float32 una sola vez para ganar velocidad
    img_float = padded.astype(np.float32)
    
    for i in range(h):
        for j in range(w):
            # Extraemos la vecindad y la aplanamos a (N, Canales)
            window = img_float[i:i+window_size, j:j+window_size].reshape(-1, c)
            out[i, j] = _truncated_vmf_logic(window, k)
            
    return out

# REFINED VECTOR MEDIAN FILTER OR DISCRETE GEOMETRIC MEDIAN APROX

from numba import jit

@jit(nopython=True)
def _refined_vmf_logic(window, search_radius):
    n = window.shape[0]
    
    # 1. VMF CLÁSICO
    dists_original = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dr = window[i, 0] - window[j, 0]
            dg = window[i, 1] - window[j, 1]
            db = window[i, 2] - window[j, 2]
            dists_original[i] += np.sqrt(dr*dr + dg*dg + db*db)
    
    best_idx = np.argmin(dists_original)
    
    # Trabajamos con escalares float64 (el estándar de Python/Numba)
    best_r = float(window[best_idx, 0])
    best_g = float(window[best_idx, 1])
    best_b = float(window[best_idx, 2])
    min_total_dist = dists_original[best_idx]
    
    # 2. REFINAMIENTO LOCAL
    r_start = max(0, int(best_r) - search_radius)
    r_end   = min(255, int(best_r) + search_radius)
    g_start = max(0, int(best_g) - search_radius)
    g_end   = min(255, int(best_g) + search_radius)
    b_start = max(0, int(best_b) - search_radius)
    b_end   = min(255, int(best_b) + search_radius)
    
    for r in range(r_start, r_end + 1):
        for g in range(g_start, g_end + 1):
            for b in range(b_start, b_end + 1):
                
                cr, cg, cb = float(r), float(g), float(b)
                current_sum = 0.0
                
                for k in range(n):
                    dr = cr - window[k, 0]
                    dg = cg - window[k, 1]
                    db = cb - window[k, 2]
                    current_sum += np.sqrt(dr*dr + dg*dg + db*db)
                
                if current_sum < min_total_dist:
                    min_total_dist = current_sum
                    best_r, best_g, best_b = cr, cg, cb
                    
    # DEVOLVEMOS UNA TUPLA: Esto evita el error de "Cannot unify array"
    return (best_r, best_g, best_b)

def refined_vector_median_filter(image, window_size, d=2):
    h, w, c = image.shape
    pad = window_size // 2
    # Convertimos la entrada a float32 para que el cálculo dentro del JIT sea fluido
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect').astype(np.float32)
    
    out = np.zeros((h, w, c), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            window = padded[i:i+window_size, j:j+window_size].reshape(-1, c)
            
            # Recibimos la tupla y Numba la asigna perfectamente a los 3 canales
            r, g, b = _refined_vmf_logic(window, d)
            out[i, j, 0] = r
            out[i, j, 1] = g
            out[i, j, 2] = b
            
    return out.astype(np.uint8)

# ==========================================
# 4. EVALUACIÓN Y EXCEL CON RESUMEN
# ==========================================

import os
import cv2
import numpy as np
import pandas as pd
# Puedes mantener o quitar el import de binary_dilation, ya no lo usaremos

def evaluate_and_finalize(edge_folder, gt_folder, excel_path, metodos_validos, tolerance=1):
    results = []
    
    gt_list = os.listdir(gt_folder)
    files = [f for f in os.listdir(edge_folder) if f.endswith('.png')]

    ### CAMBIO 1: Definir el kernel cuadrado de 3x3 (Unos en todas las direcciones)
    kernel_eval = np.ones((3, 3), np.uint8)

    print(f"Evaluando {len(files)} imágenes generadas...")
    
    metodos_encontrados = set()
    tracker = {m: {"encontrados": 0, "sin_gt": 0, "error_lectura": 0, "procesados": 0} for m in metodos_validos}
    
    metodos_validos_sorted = sorted(metodos_validos, key=len, reverse=True)

    for filename in files:
        parts = filename.split('_')
        if len(parts) < 3: continue
        
        if len(parts) >= 4:
            metodo_archivo = '_'.join(parts[3:]).replace('.png', '')
        else:
            metodo_archivo = parts[1].replace('.png', '')
            
        metodos_encontrados.add(metodo_archivo)
        
        metodo_coincidente = None
        for m in metodos_validos_sorted:
            if metodo_archivo == m or metodo_archivo.startswith(m + '_'):
                metodo_coincidente = m
                break
        
        if not metodo_coincidente: continue
            
        metodo_archivo = metodo_coincidente
        tracker[metodo_archivo]["encontrados"] += 1
        
        img_id = parts[0]
        try:
            alpha = int(parts[2])
        except ValueError: continue

        gt_file = next((f for f in gt_list if f == f"{img_id}.png" or f.endswith(f"_{img_id}.png")), None)
        if not gt_file: 
            tracker[metodo_archivo]["sin_gt"] += 1
            continue

        edge_img = cv2.imread(os.path.join(edge_folder, filename), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(os.path.join(gt_folder, gt_file), cv2.IMREAD_GRAYSCALE)
        
        if edge_img is None or gt_img is None: 
            tracker[metodo_archivo]["error_lectura"] += 1
            continue

        _, edge_img = cv2.threshold(edge_img, 127, 255, cv2.THRESH_BINARY)
        _, gt_img = cv2.threshold(gt_img, 127, 255, cv2.THRESH_BINARY)

        ### CAMBIO 2: Usar cv2.dilate con el kernel cuadrado (Tolerancia en todas direcciones)
        gt_dilated = cv2.dilate(gt_img, kernel_eval, iterations=tolerance)
        
        # Al usar OpenCV, comparamos contra 255 (blanco) en lugar de True
        tp = np.sum((edge_img == 255) & (gt_dilated == 255))
        fp = np.sum((edge_img == 255) & (gt_dilated == 0))
        
        ### CAMBIO 3: Dilatación del mapa de bordes para el cálculo de Falsos Negativos
        edge_dilated = cv2.dilate(edge_img, kernel_eval, iterations=tolerance)
        fn = np.sum((gt_img == 255) & (edge_dilated == 0))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        results.append({
            "Imagen": img_id, "Alpha": alpha, "Metodo": metodo_archivo,
            "Precision": prec, "Recall": rec, "F1-score": f1
        })
        
        tracker[metodo_archivo]["procesados"] += 1

    # ... (el resto del código del reporte y Excel se mantiene igual)
    print("\n--- REPORTE DE EVALUACIÓN POR MÉTODO ---")
    for m, stats in tracker.items():
        print(f"Método '{m}': Procesados exitosamente -> {stats['procesados']}")
    print("------------------------------------------\n")

    if not results:
        print("Atención: No se generaron resultados.")
        return pd.DataFrame() 

    df_total = pd.DataFrame(results)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for met in df_total["Metodo"].unique():
            df_met = df_total[df_total["Metodo"] == met].copy()
            df_met['F_max_imagen'] = df_met.groupby('Imagen')['F1-score'].transform('max')
            df_met = df_met.sort_values(by=["Imagen", "Alpha"])
            df_alpha_summary = df_met.groupby('Alpha')['F1-score'].mean().reset_index()
            df_alpha_summary.columns = ['Alpha_Resumen', 'F_mean_alpha']
            sheet_name = str(met)[:31]
            df_met.to_excel(writer, sheet_name=sheet_name, index=False)
            df_alpha_summary.to_excel(writer, sheet_name=sheet_name, index=False, startcol=len(df_met.columns) + 1)

    print(f"Excel final generado: {excel_path}")
    return df_total
