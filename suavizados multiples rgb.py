# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:16:20 2026

@author: user
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import time

# IMPORTANTE: Importa aquí tus funciones personalizadas si están en otro archivo
# from discrete_trunc_rgb_decomp import discrete_trunc_rgb_decomp
#%% 1. CARGANDO LAS IMÁGENES
# =============================================================================

# =============================================================================

# --- CONFIGURACIÓN DE RUTAS ---
#folder_train = 'C:/Users/user/Desktop/Datasets/BSDS500/images/train' 
folder_all = 'C:/Users/user/Desktop/Datasets/BSDS500/images/all' 
n_images = 500 # 500 images is the dataset size

images_gray = []
images_rgb = []

image_files = [f for f in os.listdir(folder_train) if f.endswith(('.jpg', '.png', '.jpeg'))]
selected_files = image_files[:min(n_images, len(image_files))]

for filename in selected_files:
    img_path = os.path.join(folder_train, filename)
    
    # Cargar en Escala de Grises
    img_g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Cargar en Color y pasar a RGB
    img_c = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_c is not None:
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    
    if img_g is not None and img_c is not None:
        images_gray.append(img_g)
        images_rgb.append(img_c)

print(f"Éxito: Se han cargado {len(images_gray)} imágenes en GRIS y {len(images_rgb)} en COLOR.")

#%% 2. CONFIGURACIÓN DEL EXPERIMENTO
# =============================================================================

# =============================================================================
# 1. Define desde qué imagen quieres empezar (0 es la primera imagen)
imag_ini = 0

# 2. Define cuántas imágenes quieres procesar en total
num_imagenes = 500 

# 3. Haz el recorte de la lista
images_a_procesar = images_rgb[imag_ini : imag_ini + num_imagenes]

# =============================================================================
# LISTA DE MÉTODOS RGB PARA EXPERIMENTO COMPLETO
# Puedes comentar (#) las líneas que no quieras ejecutar en tu prueba de 1 imagen.
# =============================================================================

metodos_a_correr = [
    # ---------------------------------------------------------
    # BLOQUE 1: MÉTODOS CLÁSICOS Y DE CONTROL
    # ---------------------------------------------------------
    #{"nombre": "NoneRGB_Clean", "func": lambda img: img, "ruido": False},
    #{"nombre": "NoneRGB_Noisy", "func": lambda img: img, "ruido": True},
    
    #{"nombre": "ClassicMeanRGB_Clean", "func": lambda img: cv2.blur(img, (3,3)), "ruido": False},
    #{"nombre": "ClassicMeanRGB_Noisy", "func": lambda img: cv2.blur(img, (3,3)), "ruido": True},
    
    #{"nombre": "ClassicMedianRGB_Clean", "func": lambda img: cv2.medianBlur(img,3), "ruido": False},
    #{"nombre": "ClassicMedianRGB_Noisy", "func": lambda img: cv2.medianBlur(img,3), "ruido": True},
    
    #{"nombre": "Gaussian_s2_RGB_Clean", "func": lambda img: cv2.GaussianBlur(img, (3,3), 1), "ruido": False},
    #{"nombre": "Gaussian_s2_RGB_Noisy", "func": lambda img: cv2.GaussianBlur(img, (3,3), 1), "ruido": True},
    
    #{"nombre": "Bilateral_RGB_Clean", "func": lambda img: cv2.bilateralFilter(img, 7, 75, 75), "ruido": False},
    #{"nombre": "Bilateral_RGB_Noisy", "func": lambda img: cv2.bilateralFilter(img, 7, 75, 75), "ruido": True},

    # ---------------------------------------------------------
    # BLOQUE 2: VECTOR MEDIAN FILTER (Exclusivo de RGB, No decomponible por definición)
    # ---------------------------------------------------------
    {"nombre": "VMF_Clean", "func": lambda img: vector_median_filter(img, 3), "ruido": False},
    {"nombre": "VMF_Noisy", "func": lambda img: vector_median_filter(img, 3), "ruido": True},
    #{"nombre": "VMFL1_Clean", "func": lambda img: vector_median_filter_L1(img, 3), "ruido": False},
    #{"nombre": "VMFL1_Noisy", "func": lambda img: vector_median_filter_L1(img, 3), "ruido": True},
    #{"nombre": "VMFtrunc_K200_Clean", "func": lambda img: truncated_vector_median_filter(img, 3, 200), "ruido": False},
    #{"nombre": "VMFtrunc_K200_Noisy", "func": lambda img: truncated_vector_median_filter(img, 3, 200), "ruido": True},
    #{"nombre": "VMFtrunc_K360_Clean", "func": lambda img: truncated_vector_median_filter(img, 3, 360), "ruido": False},
    #{"nombre": "VMFtrunc_K360_Noisy", "func": lambda img: truncated_vector_median_filter(img, 3, 360), "ruido": True},
    #{"nombre": "VMFtrunc_K420_Clean", "func": lambda img: truncated_vector_median_filter(img, 3, 420), "ruido": False},
    #{"nombre": "VMFtrunc_K420_Noisy", "func": lambda img: truncated_vector_median_filter(img, 3, 420), "ruido": True},
    #{"nombre": "VMFRefined3_Clean", "func": lambda img: refined_vector_median_filter(img, 3, 2), "ruido": False},
    #{"nombre": "VMFRefined3_Noisy", "func": lambda img: refined_vector_median_filter(img, 3, 2), "ruido": True},
    

    # ---------------------------------------------------------
    # BLOQUE 3: TRIMMED MEAN (Alpha-Trimmed d=1 REAL)
    # ---------------------------------------------------------
    #{"nombre": "Trimmed_d1_Decomp_Clean", "func": lambda img: trimmed_mean_rgb(img, 3, d=1), "ruido": False},
    #{"nombre": "Trimmed_d1_Decomp_Noisy", "func": lambda img: trimmed_mean_rgb(img, 3, d=1), "ruido": True},
    # ---------------------------------------------------------
    # BLOQUE 4: DISCRETE MEAN trunc (DMean)
    # ---------------------------------------------------------
    # K = 30
    #{"nombre": "DMean_K30_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 30, 'mean'), "ruido": False},
    #{"nombre": "DMean_K30_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 30, 'mean'), "ruido": True}, 
    # K = 90
    #{"nombre": "DMean_K90_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 90, 'mean'), "ruido": False},
    #{"nombre": "DMean_K90_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 90, 'mean'), "ruido": True},
    # K = 180
    #{"nombre": "DMean_K180_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 180, 'mean'), "ruido": False},
    #{"nombre": "DMean_K180_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 180, 'mean'), "ruido": True},
    # K = 250
    #{"nombre": "DMean_K250_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 250, 'mean'), "ruido": False},
    #{"nombre": "DMean_K250_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 250, 'mean'), "ruido": True},
  
    # ---------------------------------------------------------
    # BLOQUE 5: DISCRETE MEDIAN trunc (DMedian)
    # ---------------------------------------------------------
    # K = 10
    #{"nombre": "DMedian_K10_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 10, 'median'), "ruido": False},
    #{"nombre": "DMedian_K10_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 10, 'median'), "ruido": True},
    # K = 20
    #{"nombre": "DMedian_K20_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 20, 'median'), "ruido": False},
    #{"nombre": "DMedian_K20_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 20, 'median'), "ruido": True},
    # K = 30
    #{"nombre": "DMedian_K30_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 30, 'median'), "ruido": False},
    #{"nombre": "DMedian_K30_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 30, 'median'), "ruido": True},
    # K = 90
    #{"nombre": "DMedian_K90_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 90, 'median'), "ruido": False},
    #{"nombre": "DMedian_K90_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 90, 'median'), "ruido": True},
    # K = 180
    #{"nombre": "DMedian_K180_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 180, 'median'), "ruido": False},
    #{"nombre": "DMedian_K180_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 180, 'median'), "ruido": True},
    # K = 250
    #{"nombre": "DMedian_K250_Decomp_Clean", "func": lambda img: discrete_trunc_decomp(img, 3, 250, 'median'), "ruido": False},
    #{"nombre": "DMedian_K250_Decomp_Noisy", "func": lambda img: discrete_trunc_decomp(img, 3, 250, 'median'), "ruido": True},
]

outputfolder = 'C:/Users/user/Desktop/Research/Agregaciones Direccionales/Imagenes/BinRGBall'
ground_truth_folder = 'C:/Users/user/Desktop/Datasets/BSDS500/groundTruth/all'
output_excel = "C:/Users/user/Desktop/Research/Agregaciones Direccionales/resultados/Evaluacion_RGBall.xlsx"

os.makedirs(outputfolder, exist_ok=True)
thrs = list(range(20, 241, 20))

#%% 3. BUCLE PRINCIPAL DE PROCESAMIENTO
# =============================================================================

print(f"Iniciando procesamiento RGB ({len(images_a_procesar)} imágenes)...")

for config in metodos_a_correr:
    nombre_metodo = config["nombre"]
    usa_ruido = config.get("ruido", False)
    t_metodo = 0 
    
    for i, img_raw in enumerate(images_a_procesar):
        inicio_img = time.time()
        image_name = os.path.splitext(selected_files[i])[0] # Ojo: usar selected_files aquí
        
        img_pre = sal_y_pimienta(img_raw, 0.05) if usa_ruido else img_raw
        
        try:
            if "k" in config:
                filtered = config["func"](img_pre, 3, config["k"])
            else:
                filtered = config["func"](img_pre)
        except Exception as e:
            print(f"ERROR en {nombre_metodo} con {image_name}: {e}")
            continue
            
        for thr in thrs:
            edges = sobel_universal(filtered, thr, 3)
            filename = f"{image_name}_Sobel_{thr}_{nombre_metodo}.png"
            cv2.imwrite(os.path.join(outputfolder, filename), edges)
            
        duracion = time.time() - inicio_img
        t_metodo += duracion
        print(f"[{nombre_metodo}] {i+1}/{num_imagenes} ('{image_name}') -> {duracion:.2f} s.")
    
    print(f"==> MÉTODO {nombre_metodo} FINALIZADO en {t_metodo/60:.2f} min.")
#%% 4. EVALUACIÓN Y EXCEL CON RESUMEN
# ==========================================
    
from scipy.ndimage import binary_dilation

# Crea la lista con los nombres exactos que quieres evaluar
mis_metodos_a_evaluar_rgb = [
    # --- BLOQUE 1: Clásicos y Control ---
    'NoneRGB_Clean', 'NoneRGB_Noisy',
    'ClassicMeanRGB_Clean', 'ClassicMeanRGB_Noisy',
    'ClassicMedianRGB_Clean', 'ClassicMedianRGB_Noisy',
    'Gaussian_s2_RGB_Clean', 'Gaussian_s2_RGB_Noisy',
    'Bilateral_RGB_Clean', 'Bilateral_RGB_Noisy',

    # --- BLOQUE 2: Vector Median Filter (VMF) ---
    'VMF_Clean', 'VMF_Noisy',

    # --- BLOQUE 3: Trimmed Mean ---
    'Trimmed_d1_Decomp_Clean', 'Trimmed_d1_Decomp_Noisy',

    # --- BLOQUE 4: Discrete Mean (DMean) ---
    #'DMean_K30_Decomp_Clean', 'DMean_K30_Decomp_Noisy',
    'DMean_K90_Decomp_Clean', 'DMean_K90_Decomp_Noisy',
    'DMean_K180_Decomp_Clean', 'DMean_K180_Decomp_Noisy',
    'DMean_K250_Decomp_Clean', 'DMean_K250_Decomp_Noisy',

    # --- BLOQUE 5: Discrete Median (DMedian) ---
    'DMedian_K90_Decomp_Clean', 'DMedian_K90_Decomp_Noisy',
    'DMedian_K180_Decomp_Clean', 'DMedian_K180_Decomp_Noisy',
    'DMedian_K250_Decomp_Clean', 'DMedian_K250_Decomp_Noisy'
]

mis_metodos_a_evaluar_Median = ['ClassicMedianRGB_Clean', 'ClassicMedianRGB_Noisy']
#mis_metodos_a_evaluar_VMF = ['VMFRefined3_Clean', 'VMFRefined3_Noisy']

# --- Ejecución ---
# Así se ejecutará, te generará el Excel, y además guardará los datos en la variable df_resultados
df_resultados_RGB = evaluate_and_finalize(outputfolder, ground_truth_folder, output_excel, mis_metodos_a_evaluar_rgb, tolerance=1)

df_resultados_Median = evaluate_and_finalize(outputfolder, ground_truth_folder, output_excel, mis_metodos_a_evaluar_Median, tolerance=1)
# Para comprobar que funciona, puedes imprimir las primeras filas en la consola:
print(df_resultados_RGB.head())

#%% 5. ANALIZAR Y GRAFICAR
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def clasificar_metodo(nombre_metodo):
    """
    Identifica la versión (Clean o Noisy) y el método base a partir del nombre original.
    Funciona tanto con la nomenclatura de grises como la de RGB.
    """
    if 'Noisy-' in nombre_metodo:
        return nombre_metodo.replace('Noisy-', ''), 'Noisy'
    elif '_Noisy' in nombre_metodo:
        return nombre_metodo.replace('_Noisy', ''), 'Noisy'
    elif '_Clean' in nombre_metodo:
        return nombre_metodo.replace('_Clean', ''), 'Clean'
    else:
        # Por descarte, los clásicos de grises que no tienen sufijo son la versión limpia
        return nombre_metodo, 'Clean'

def analizar_y_graficar_avanzado(df_resultados, titulo_general):
    """
    Genera las tablas LaTeX, los Boxplots de Fmax (Clean/Noisy/Hybrid) 
    y los gráficos de líneas separados para Clean y Noisy.
    """
    # 1. Limpieza de duplicados
    df_clean = df_resultados.drop_duplicates(subset=['Imagen', 'Alpha', 'Metodo'], keep='last').copy()
    df_clean['Alpha'] = pd.to_numeric(df_clean['Alpha'])
    df_clean['F1-score'] = pd.to_numeric(df_clean['F1-score'])

    # 2. Asignar Base y Condición
    df_clean[['Base_Metodo', 'Condicion']] = pd.DataFrame(
        df_clean['Metodo'].apply(clasificar_metodo).tolist(), 
        index=df_clean.index
    )

    print(f"\n{'='*60}")
    print(f" RESULTADOS: {titulo_general}")
    print(f"{'='*60}\n")

    # ==========================================
    # PREPARACIÓN DE DATOS (F_MAX POR IMAGEN)
    # ==========================================
    # Calculamos el max F1-score por cada imagen y método exacto
    df_fmax = df_clean.groupby(['Base_Metodo', 'Condicion', 'Imagen'])['F1-score'].max().reset_index()
    
    # Pivotamos para tener columnas Clean y Noisy en la misma fila por imagen
    df_pivot = df_fmax.pivot(index=['Base_Metodo', 'Imagen'], columns='Condicion', values='F1-score').reset_index()
    
    # Rellenamos posibles nulos (si un método falta en alguna condición) con 0
    if 'Clean' not in df_pivot.columns: df_pivot['Clean'] = 0.0
    if 'Noisy' not in df_pivot.columns: df_pivot['Noisy'] = 0.0
    df_pivot.fillna({'Clean': 0.0, 'Noisy': 0.0}, inplace=True)

    # Creamos la métrica Híbrida (ponderada al 50%) para cada imagen
    df_pivot['Hybrid'] = 0.5 * df_pivot['Clean'] + 0.5 * df_pivot['Noisy']

    # Derretimos de nuevo el DataFrame para que Seaborn pueda hacer el boxplot
    df_boxplot = df_pivot.melt(id_vars=['Base_Metodo', 'Imagen'], 
                               value_vars=['Clean', 'Noisy', 'Hybrid'],
                               var_name='Escenario', value_name='F_max')

    # Para ordenar el boxplot de mayor a menor por el Híbrido Promedio
    orden_metodos = df_pivot.groupby('Base_Metodo')['Hybrid'].mean().sort_values(ascending=False).index

    # ==========================================
    # DATOS PARA TABLA 2 (LATEX) - F_max Global
    # ==========================================
    print("--- DATOS TABLA 2 (F1-score Máximo Promedio Global) ---")
    df_global = df_pivot.groupby('Base_Metodo')[['Clean', 'Noisy', 'Hybrid']].mean().loc[orden_metodos]
    
    for base, row in df_global.iterrows():
        nombre_latex = str(base).replace('_', '\\_')
        print(f"{nombre_latex} & {row['Clean']:.4f} & {row['Noisy']:.4f} & {row['Hybrid']:.4f} \\\\")

    # ==========================================
    # GRÁFICA 1: BOXPLOT DE F_MAX (3 CATEGORÍAS)
    # ==========================================
    plt.figure(figsize=(16, 8))
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # Paleta de colores personalizada (Azul=Clean, Rojo=Noisy, Morado=Híbrido)
    paleta_boxplot = {'Clean': '#2ecc71', 'Noisy': '#e74c3c', 'Hybrid': '#9b59b6'}
    
    ax = sns.boxplot(data=df_boxplot, x='Base_Metodo', y='F_max', hue='Escenario', 
                     order=orden_metodos, palette=paleta_boxplot, showfliers=False)
    
    plt.title(f'Maximum F1-score (400 images) - {titulo_general}', fontweight='bold')
    plt.xlabel('Smoothing method')
    plt.ylabel('Global Maximum F1-score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Modality')
    plt.tight_layout()
    plt.show()

    # ==========================================
    # GRÁFICA 2: LÍNEAS (CLEAN vs NOISY)
    # ==========================================
    # Calcular promedios por Alpha para dibujar las líneas
    df_lineas = df_clean.groupby(['Metodo', 'Condicion', 'Alpha'])['F1-score'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig.suptitle(f'F1-score average vs Alpha threshold - {titulo_general}', fontsize=16, fontweight='bold')

    # Subplot 1: Clean
    sns.lineplot(data=df_lineas[df_lineas['Condicion'] == 'Clean'], 
                 x='Alpha', y='F1-score', hue='Metodo', ax=axes[0], marker='o', linewidth=2)
    axes[0].set_title('Clean images scenario', fontsize=14)
    axes[0].set_xlabel('Alpha threshold')
    axes[0].set_ylabel('F1-score Mean')
    axes[0].set_xticks(sorted(df_clean['Alpha'].unique()))
    axes[0].legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=2)

    # Subplot 2: Noisy
    sns.lineplot(data=df_lineas[df_lineas['Condicion'] == 'Noisy'], 
                 x='Alpha', y='F1-score', hue='Metodo', ax=axes[1], marker='o', linewidth=2)
    axes[1].set_title('Noisy images scenario', fontsize=14)
    axes[1].set_xlabel('Alpha threshold')
    axes[1].set_xticks(sorted(df_clean['Alpha'].unique()))
    axes[1].legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Ajuste para que la leyenda no se corte
    plt.show()

# --- EJECUCIÓN DEL SCRIPT ---

# 2. Analizar RGB
analizar_y_graficar_avanzado(df_resultados_Median, "RGB_Median Images")
np.save('C:/Users/user/Desktop/Research/Agregaciones Direccionales/resultados/df_resultados_RGBMedian', df_resultados_Median)

#%% WILCOXON

import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

def generar_tabla_wilcoxon_global_rgb(df_resultados, pares_a_comparar, titulo="Comparativas Wilcoxon Global RGB"):
    """
    Versión para RGB: Une Clean y Noisy y limpia nombres complejos de métodos.
    """
    df_proc = df_resultados.copy()
    
    # Asegurar que el F1 sea numérico
    df_proc['F1-score'] = pd.to_numeric(df_proc['F1-score'])

    # FUNCIÓN DE LIMPIEZA MEJORADA PARA RGB
    def limpiar_nombre_metodo_rgb(nombre):
        # Eliminamos indicadores de ruido/limpio y sufijos de implementación
        for tag in ['Noisy-', 'Noisy_', '_Noisy', '_Clean', 'Clean_', '_Decomp', '_RGB', 'RGB_']:
            nombre = nombre.replace(tag, '')
        return nombre.strip('_').strip()

    df_proc['Metodo_Base'] = df_proc['Metodo'].apply(limpiar_nombre_metodo_rgb)
    
    # Clasificación simple para separar las muestras pareadas
    df_proc['Condicion'] = df_proc['Metodo'].apply(lambda x: 'Noise' if 'Nois' in x else 'Clean')

    # Pivotar para tener una columna por método base
    df_fmax = df_proc.groupby(['Imagen', 'Condicion', 'Metodo_Base'])['F1-score'].max().unstack('Metodo_Base')

    resultados_test = []

    for metA, metB in pares_a_comparar:
        # Limpiamos los nombres que pases en la lista de pares
        metA_L = limpiar_nombre_metodo_rgb(metA)
        metB_L = limpiar_nombre_metodo_rgb(metB)

        if metA_L not in df_fmax.columns or metB_L not in df_fmax.columns:
            print(f" Advertencia: {metA_L} o {metB_L} no encontrados. Se omite.")
            continue
            
        data = df_fmax[[metA_L, metB_L]].dropna()
        distA = data[metA_L]
        distB = data[metB_L]
        
        try:
            stat, p_value = wilcoxon(distA, distB, zero_method='wilcox')
        except ValueError:
            p_value = 1.0 
            
        diferencia_media = (distA - distB).mean()
        
        if p_value < 0.001: sig = '***'
        elif p_value < 0.01: sig = '**'
        elif p_value < 0.05: sig = '*'
        else: sig = 'n.s.'
            
        ganador = "Empate"
        if sig != 'n.s.':
            ganador = metA_L if diferencia_media > 0 else metB_L

        resultados_test.append({
            'Metodo_A': metA_L, 'Metodo_B': metB_L,
            'Diff_Media': diferencia_media, 'P_value': p_value,
            'Sig': sig, 'Ganador': ganador, 'N': len(data)
        })

    # GENERACIÓN DE LATEX
    print(f"\n% --- TABLA LATEX: {titulo} ---")
    print("\\begin{table}[htbp]\n\\centering")
    print("\\caption{Global Wilcoxon test (Clean + Noisy) for RGB images. samples: N=" + str(resultados_test[0]['N'] if resultados_test else 0) + "}")
    print("\\begin{tabular}{llcccl}\n\\toprule")
    print("\\textbf{Method A} & \\textbf{Method B} & \\textbf{$\\Delta$ Mean} & \\textbf{$p$-value} & \\textbf{Sig.} & \\textbf{Advantage} \\\\\n\\midrule")
    
    for res in resultados_test:
        metA_tex = res['Metodo_A'].replace('_', '\\_')
        metB_tex = res['Metodo_B'].replace('_', '\\_')
        p_str = "$< 0.001$" if res['P_value'] < 0.001 else f"{res['P_value']:.4f}"
        diff_str = f"{'+' if res['Diff_Media'] > 0 else ''}{res['Diff_Media']:.4f}"
        ganador_str = res['Ganador'].replace('_', '\\_')
        if res['Ganador'] != "Empate": ganador_str = f"\\textbf{{{ganador_str}}}"
        
        print(f"{metA_tex} & {metB_tex} & {diff_str} & {p_str} & {res['Sig']} & {ganador_str} \\\\")
        
    print("\\bottomrule\n\\end{tabular}\n\\end{table}")

# --- LISTA DE PARES GLOBAL PARA RGB ---
# Aquí pones los nombres "base" o completos, la función los limpiará.
pares_globales_rgb = [
    ('DMean_K90', 'Gaussian_s2'),
    ('DMean_K90', 'Bilateral'),
    ('DMean_K180', 'ClassicMean'),
    ('DMedian_K250', 'ClassicMedian'),
    ('DMean_K90', 'DMedian_K30')
]

# Llamada
generar_tabla_wilcoxon_global_rgb(df_resultados_RGB, pares_globales_rgb)

#%% 7. TABLAS PROMEDIOS FMAX
import pandas as pd
import numpy as np

def generar_tabla_promedios_fmax(df, titulo_tabla, label_latex, num_imagenes=100):
    """
    Procesa el DataFrame para obtener los promedios de los F-máximos y genera código LaTeX.
    """
    # 1. Limpieza de nombres mejorada
    def clasificar(row):
        metodo = row['Metodo']
        # Identificar escenario
        if 'Noisy' in metodo or 'noisy' in metodo.lower():
            escenario = 'Noisy'
        else:
            escenario = 'Clean'
        
        # LIMPIEZA AGRESIVA PARA UNIFICAR:
        # Quitamos Clean/Noisy y también la palabra RGB para que 'ClassicMedianRGB' sea igual a 'ClassicMedian'
        nombre_base = metodo.replace('Noisy-', '').replace('_Noisy', '').replace('Noisy_', '')
        nombre_base = nombre_base.replace('_Clean', '').replace('Clean_', '')
        nombre_base = nombre_base.replace('RGB', '') # <--- Esto unifica los casos que mencionas
        
        # Eliminar guiones bajos sobrantes al final para estética
        nombre_base = nombre_base.strip('_')
        
        return nombre_base, escenario

    df_prep = df.copy()
    # Aplicamos la clasificación
    df_prep[['Metodo_Base', 'Escenario']] = df_prep.apply(lambda r: pd.Series(clasificar(r)), axis=1)
    
    # 2. Encontrar el F1-score máximo por Imagen, Método y Escenario (el mejor Alpha)
    fmax_por_imagen = df_prep.groupby(['Imagen', 'Metodo_Base', 'Escenario'])['F1-score'].max().reset_index()
    
    # 3. Calcular el promedio de esos máximos
    # Usamos dropna=False para ver qué está pasando si algo falla
    promedios = fmax_por_imagen.groupby(['Metodo_Base', 'Escenario'])['F1-score'].mean().unstack()
    
    # 4. Calcular el TOTAL Average ignorando NaNs (si algún método solo existe en un escenario)
    if 'Clean' in promedios.columns and 'Noisy' in promedios.columns:
        promedios['TOTAL Average'] = promedios[['Clean', 'Noisy']].mean(axis=1)
    else:
        promedios['TOTAL Average'] = promedios.mean(axis=1)
        
    promedios = promedios.sort_values(by='TOTAL Average', ascending=False)

    # 5. GENERACIÓN DE CÓDIGO LATEX
    print(f"\n% --- TABLA LATEX: {titulo_tabla} ---")
    print("\\begin{table*}[htbp]")
    print("\\centering")
    print("\\color{blue}")
    print(f"\\caption{{{titulo_tabla} ($N = {num_imagenes}$ images analyzed).}}")
    print(f"\\label{{{label_latex}}}")
    print("\\begin{adjustbox}{width=0.85\\textwidth}")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("\\textbf{Smoothing Method} & \\textbf{Clean Images} & \\textbf{Noisy Images} & \\textbf{TOTAL Average} \\\\")
    print("\\midrule")
    
    for metodo, row in promedios.iterrows():
        # Lógica de negritas para tus métodos (añadimos Trunc y DMean/DMedian)
        es_nuestro = any(x in metodo for x in ['DMean', 'DMedian', 'Trunc', 'Trimmed'])
        
        # Formateo de nombres para la tabla
        display_name = metodo
        if 'DMean' in display_name: display_name = display_name.replace('DMean', 'TruncL2 ') + " (Ours)"
        if 'DMedian' in display_name: display_name = display_name.replace('DMedian', 'TruncL1 ') + " (Ours)"
        if 'ClassicMedian' in display_name: display_name = 'Classical Median'
        if 'ClassicMean' in display_name: display_name = 'Classical Mean'
        if 'None' in display_name: display_name = 'No Smoothing (None)'

        # Gestión de valores para evitar imprimir "nan"
        c_val = f"{row['Clean']:.4f}" if ('Clean' in row and not pd.isna(row['Clean'])) else "N/A"
        n_val = f"{row['Noisy']:.4f}" if ('Noisy' in row and not pd.isna(row['Noisy'])) else "N/A"
        t_val = f"{row['TOTAL Average']:.4f}" if not pd.isna(row['TOTAL Average']) else "N/A"
        
        if es_nuestro:
            linea = f"\\textbf{{{display_name}}} & \\textbf{{{c_val}}} & \\textbf{{{n_val}}} & \\textbf{{{t_val}}} \\\\"
        else:
            linea = f"{display_name} & {c_val} & {n_val} & {t_val} \\\\"
            
        print(linea)
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{adjustbox}")
    print("\\end{table*}")

# --- EJECUCIÓN ---
# Generar tabla para Grises (suponiendo N=200 imágenes)
generar_tabla_promedios_fmax(df_resultados_gray, "Global F1-score evaluation (Grayscale)", "tab:results_gray", num_imagenes=200)

# Generar tabla para RGB (suponiendo N=100 imágenes)
generar_tabla_promedios_fmax(df_resultados_RGB, "Global F1-score evaluation (RGB)", "tab:results_rgb", num_imagenes=200)

# Genera la tabla para otros métodos puntuales (VMF, etc.)
generar_tabla_promedios_fmax(df_resultados_VMF, "Global F1-score evaluation (RGB)", "tab:results_VMF", num_imagenes=200)
