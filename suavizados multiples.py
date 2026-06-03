import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
import time

# 0. CARGANDO LAS IMÁGENES

import os
import cv2

# --- CONFIGURACIÓN DE RUTAS ---
#folder_train = 'C:/Users/user/Desktop/Datasets/BSDS500/images/train' 
folder_all = 'C:/Users/user/Desktop/Datasets/BSDS500/images/all' 
n_images = 500 # 500 images is the dataset size

# Listas para almacenar las imágenes
images_gray = []
images_rgb = []

# Obtener lista de archivos
image_files = [f for f in os.listdir(folder_all) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Seleccionamos las primeras N imágenes
selected_files = image_files[:min(n_images, len(image_files))]

for filename in selected_files:
    img_path = os.path.join(folder_all, filename)
    
    # 1. Cargar en Escala de Grises (2D: Alto x Ancho)
    img_g = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. Cargar en Color (3D: Alto x Ancho x Canales)
    # OpenCV carga por defecto en BGR, lo convertimos a RGB para que 
    # tus métodos 'nondecomp' funcionen con el orden correcto.
    img_c = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
    
    if img_g is not None and img_c is not None:
        images_gray.append(img_g)
        images_rgb.append(img_c)

print(f"Éxito: Se han cargado {len(images_gray)} imágenes en GRIS y {len(images_rgb)} en COLOR.")
print(f"Formato Gris: {images_gray[0].shape}") # Debería ser (H, W)
print(f"Formato RGB: {images_rgb[0].shape}")   # Debería ser (H, W, 3)

#%%
# ==========================================
# CONFIGURACIÓN DEL EXPERIMENTO
# ==========================================

# 1. Define desde qué imagen quieres empezar (0 es la primera imagen)
imag_ini = 0

# 2. Define cuántas imágenes quieres procesar en total
num_imagenes = 500

# 3. Haz el recorte de la lista
images_a_procesar = images_gray[imag_ini : imag_ini + num_imagenes]

import cv2
import numpy as np

metodos_a_correr = [
    # --- Discrete Mean Truncated ---
    {"nombre": "DMean90", "k": 90, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": False},
    {"nombre": "DMean180", "k": 180, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": False},
    {"nombre": "DMean250", "k": 250, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": False},
    
    # --- Noisy Discrete Mean ---
    {"nombre": "Noisy-DMean90", "k": 90, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": True},
    {"nombre": "Noisy-DMean180", "k": 180, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": True},
    {"nombre": "Noisy-DMean250", "k": 250, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='mean'), "ruido": True},

    # --- Discrete Median Truncated ---
    {"nombre": "DMedian90", "k": 90, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": False},
    {"nombre": "Noisy-DMedian90", "k": 90, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": True},
    {"nombre": "DMedian180", "k": 180, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": False},
    {"nombre": "Noisy-DMedian180", "k": 180, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": True},
    {"nombre": "DMedian250", "k": 250, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": False},
    {"nombre": "Noisy-DMedian250", "k": 250, "func": lambda img, w, k: discrete_trunc_decomp(img, w, k, mode='median'), "ruido": True},

    #---Trimmed        
    {"nombre": "Trimmed_d1_Decomp_Clean", "func": lambda img: trimmed_mean_rgb(img, kernel_size=3, d=1), "ruido": False},
    {"nombre": "Trimmed_d1_Decomp_Noisy", "func": lambda img: trimmed_mean_rgb(img, kernel_size=3, d=1), "ruido": True},

    # --- Filtros Clásicos 
    {"nombre": "ClassicMean", "func": lambda img: cv2.blur(img, (3,3)), "ruido": False},
    {"nombre": "Noisy-ClassicMean", "func": lambda img: cv2.blur(img, (3,3)), "ruido": True},
    {"nombre": "Bilateral", "func": lambda img: cv2.bilateralFilter(img, 7, 75, 75), "ruido": False},
    {"nombre": "Noisy-Bilateral", "func": lambda img: cv2.bilateralFilter(img, 7, 75, 75), "ruido": True},
    {"nombre": "Gaussian_s2", "func": lambda img: cv2.GaussianBlur(img, (3,3), 1), "ruido": False},
    {"nombre": "Noisy-Gaussian_s2", "func": lambda img: cv2.GaussianBlur(img, (3,3), 1), "ruido": True},
    {"nombre": "ClassicMedian", "func": lambda img: cv2.medianBlur(img,3), "ruido": False},
    {"nombre": "Noisy-ClassicMedian", "func": lambda img: cv2.medianBlur(img,3), "ruido": True},
    
    # --- NON-SMOOTHING (Base para comparar) ---
    {"nombre": "Nonsmoothing", "func": lambda img: img, "ruido": False},
    {"nombre": "Nonsmoothing_Noisy", "func": lambda img: img, "ruido": True},
]

outputfolder = 'C:/Users/user/Desktop/Research/Agregaciones Direccionales/Imagenes/BinGrayAll'
ground_truth_folder = 'C:/Users/user/Desktop/Datasets/BSDS500/groundTruth/all'

os.makedirs(outputfolder, exist_ok=True)
thrs = list(range(20, 241, 20)) # Lista de Alphas (12 valores)
tolerance = 1 

# Lista de nombres permitidos para filtrar archivos viejos en la evaluación
nombres_metodos_actuales = [m["nombre"] for m in metodos_a_correr]

# ==========================================
# 3. BUCLE DE GENERACIÓN DE IMÁGENES
# ==========================================

print("Iniciando procesamiento...")
tiempo_inicio_total = time.time() # Para saber cuánto tarda TODO el script

for config in metodos_a_correr:
    nombre_metodo = config["nombre"]
    usa_ruido = config.get("ruido", False)
    tiempo_acumulado_metodo = 0 # Reiniciamos para cada método
    
    for i, img_gray in enumerate(images_a_procesar):
        inicio_img = time.time()
        image_name = os.path.splitext(image_files[i])[0]
        
        # Procesar
        img_pre = sal_y_pimienta(img_gray, 0.05) if usa_ruido else img_gray
        
        # Llamada dinámica a la función (usa la que esté en memoria)
        if "k" in config:
            filtered = config["func"](img_pre, 3, config["k"])
        else:
            filtered = config["func"](img_pre)
            
        # Alphas
        for thr in thrs:
            edges = sobel_universal(filtered, thr, 3)
            filename = f"{image_name}_Sobel_{thr}_{nombre_metodo}.png"
            cv2.imwrite(os.path.join(outputfolder, filename), edges)
        
        duracion_img = time.time() - inicio_img
        tiempo_acumulado_metodo += duracion_img
        
        print(f"[{nombre_metodo}] Imagen {i+1}/{num_imagenes} ('{image_name}') -> {duracion_img:.2f} s.")
    
    # --- ACLARACIÓN DE TIEMPOS ---
    # Convertimos segundos a formato Minutos:Segundos para que no sea un "número extraño"
    mins, secs = divmod(tiempo_acumulado_metodo, 60)
    print(f"==> MÉTODO {nombre_metodo} FINALIZADO. Tiempo total (todas las imágenes): {int(mins)}min {secs:.2f}s.")
    
#%%
# ==========================================
# 4. EVALUACIÓN Y EXCEL CON RESUMEN
# ==========================================
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

output_excel = "C:/Users/user/Desktop/Research/Agregaciones Direccionales/resultados/Evaluacion_Gray.xlsx"

# --- Ejecución ---
# Crea la lista con los nombres exactos que quieres evaluar. Deben coincidir con los nombres de las imágenes binarizadas
mis_metodos_a_evaluar = [
    #--- Discrete Mean Truncated ---
    'DMean90', 'DMean180', 'DMean250',
    'Noisy-DMean90', 'Noisy-DMean180', 'Noisy-DMean250',

    #--- Discrete Median Truncated ---
    'DMedian90', 'Noisy-DMedian90',
    'DMedian180', 'Noisy-DMedian180',
    'DMedian250', 'Noisy-DMedian250',

    # --- Trimmed ---
    'Trimmed_d1_Decomp_Clean', 'Trimmed_d1_Decomp_Noisy',

    # --- Filtros Clásicos ---
    'ClassicMean', 'Noisy-ClassicMean',
    'ClassicMedian', 'Noisy-ClassicMedian', 
    'Bilateral', 'Noisy-Bilateral',         
    'Gaussian_s2', 'Noisy-Gaussian_s2',

    # --- Non-Smoothing ---
    'Nonsmoothing', 'Nonsmoothing_Noisy'
]

#para comprobar posibles fallos
import os
print(f"Buscando en: {outputfolder}")
if os.path.exists(outputfolder):
    archivos_png = [f for f in os.listdir(outputfolder) if f.endswith('.png')]
    print(f"¡Encontrados {len(archivos_png)} archivos .png!")
else:
    print("LA CARPETA NO EXISTE EN ESA RUTA. Revisa el texto.")

df_resultados_gray = evaluate_and_finalize(outputfolder, ground_truth_folder, output_excel, mis_metodos_a_evaluar, tolerance=1)

#%%ANALIZAR Y GRAFICAR DATA FRAME
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
    
    plt.title(f'Maximum F1-score (1000 images) - {titulo_general}', fontweight='bold')
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
    fig.suptitle(f'F1-score average evolution vs Alpha threshold - {titulo_general}', fontsize=16, fontweight='bold')

    # Subplot 1: Clean
    sns.lineplot(data=df_lineas[df_lineas['Condicion'] == 'Clean'], 
                 x='Alpha', y='F1-score', hue='Metodo', ax=axes[0], marker='o', linewidth=2)
    axes[0].set_title('Clean images scenario', fontsize=14)
    axes[0].set_xlabel('Alpha threshold')
    axes[0].set_ylabel('F1-score average')
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
# 1. Analizar escala de grises
analizar_y_graficar_avanzado(df_resultados_gray, "Grayscale")
np.save('C:/Users/user/Desktop/Research/Agregaciones Direccionales/resultados/df_resultados_gray2', df_resultados_gray)


#%% WILCOXON
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np

def generar_tabla_wilcoxon_global(df_resultados, pares_a_comparar, titulo="Comparativas Wilcoxon Global"):
    """
    Realiza el test de Wilcoxon unificando Clean y Noisy para cada método.
    """
    df_proc = df_resultados.copy()
    
    # 1. Estandarizar nombres de columnas (por si acaso varían entre versiones de pandas)
    # Asumimos que las columnas son ['Imagen', 'Threshold', 'Metodo', 'F1-score']
    df_proc['F1-score'] = pd.to_numeric(df_proc['F1-score'])

    # 2. FUNCIÓN PARA LIMPIAR EL NOMBRE DEL MÉTODO
    # Esto une 'DMean90' y 'Noisy-DMean90' en una sola categoría: 'DMean90'
    def limpiar_nombre_metodo(nombre):
        for prefijo in ['Noisy-', 'Noisy_', '_Noisy', '_Clean', 'Clean_']:
            nombre = nombre.replace(prefijo, '')
        return nombre.strip()

    df_proc['Metodo_Base'] = df_proc['Metodo'].apply(limpiar_nombre_metodo)
    
    # Identificamos si la fila original era Clean o Noisy para no colapsar datos erróneamente
    df_proc['Condicion'] = df_proc['Metodo'].apply(lambda x: 'Noise' if 'Nois' in x else 'Clean')

    # 3. Obtener el F_max para cada terna (Imagen, Condicion, Metodo_Base)
    # Queremos el mejor F1 que dio ese método para esa imagen en esa circunstancia
    df_fmax = df_proc.groupby(['Imagen', 'Condicion', 'Metodo_Base'])['F1-score'].max().unstack('Metodo_Base')

    resultados_test = []

    for metA, metB in pares_a_comparar:
        # Limpiamos también los nombres de los pares por si se pasan con "Noisy-"
        metA_limpio = limpiar_nombre_metodo(metA)
        metB_limpio = limpiar_nombre_metodo(metB)

        if metA_limpio not in df_fmax.columns or metB_limpio not in df_fmax.columns:
            print(f" Advertencia: {metA_limpio} o {metB_limpio} no encontrados. Se omite.")
            continue
            
        # Extraemos las distribuciones (tendrán el doble de tamaño: N_imagenes * 2)
        data = df_fmax[[metA_limpio, metB_limpio]].dropna()
        distA = data[metA_limpio]
        distB = data[metB_limpio]
        
        # Test de Wilcoxon
        try:
            stat, p_value = wilcoxon(distA, distB, zero_method='wilcox')
        except ValueError:
            p_value = 1.0 
            
        diferencia_media = (distA - distB).mean()
        
        if p_value < 0.001: sig = '***'
        elif p_value < 0.01: sig = '**'
        elif p_value < 0.05: sig = '*'
        else: sig = 'n.s.'
            
        if sig != 'n.s.':
            ganador = metA_limpio if diferencia_media > 0 else metB_limpio
        else:
            ganador = "Empate"

        resultados_test.append({
            'Metodo_A': metA_limpio,
            'Metodo_B': metB_limpio,
            'Diff_Media': diferencia_media,
            'P_value': p_value,
            'Sig': sig,
            'Ganador': ganador,
            'N_muestras': len(data)
        })

    # ==========================================
    # GENERACIÓN DE LA TABLA EN LATEX
    # ==========================================
    print(f"\n% --- TABLA LATEX: {titulo} ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Wilcoxon signed-rank test (Global: Clean + Noisy). N=" + str(resultados_test[0]['N_muestras'] if resultados_test else 0) + " samples.}")
    print("\\begin{tabular}{llcccl}")
    print("\\toprule")
    print("\\textbf{Method A} & \\textbf{Method B} & \\textbf{$\\Delta$ Mean} & \\textbf{$p$-value} & \\textbf{Sig.} & \\textbf{Advantage} \\\\")
    print("\\midrule")
    
    for res in resultados_test:
        metA_tex = res['Metodo_A'].replace('_', '\\_')
        metB_tex = res['Metodo_B'].replace('_', '\\_')
        p_str = "$< 0.001$" if res['P_value'] < 0.001 else f"{res['P_value']:.4f}"
        diff_str = f"{'+' if res['Diff_Media'] > 0 else ''}{res['Diff_Media']:.4f}"
        ganador_str = res['Ganador'].replace('_', '\\_')
        if res['Ganador'] != "Empate":
            ganador_str = f"\\textbf{{{ganador_str}}}"
        
        print(f"{metA_tex} & {metB_tex} & {diff_str} & {p_str} & {res['Sig']} & {ganador_str} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

# --- EJEMPLO DE USO ---
# Ahora los pares son simplemente el "nombre base" del método
pares_globales = [
    ('DMean90', 'Gaussian_s2'),
    ('DMean180', 'ClassicMean'),
    ('DMedian250', 'ClassicMedian'),
    ('DMean90', 'Bilateral'),
    ('DMean90', 'DMedian90')
]

# Llamada a la función
generar_tabla_wilcoxon_global(df_resultados_gray, pares_globales, "Comparativa Global (Clean + Noisy)")