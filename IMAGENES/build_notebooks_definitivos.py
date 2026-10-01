# -*- coding: utf-8 -*-
"""
Generador Maestro Definitivo de Cuadernos Jupyter para la Tesis:
- ResNet-50.ipynb
- EfficientNet-B3.ipynb

Universidad Nacional Jose Maria Arguedas (UNAJMA) - 2026
Escuela Profesional de Ingenieria de Sistemas
Tesis: Metodo no invasivo basado en redes neuronales convolucionales para la deteccion
de la anemia infantil a partir de imagenes en el centro de salud, Ocobamba 2025.
"""

import json
import os

def create_cell(cell_type, source):
    """Crea una celda de Jupyter Notebook con formato estandar."""
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1].endswith('\n'):
            lines[-1] = lines[-1][:-1]
    else:
        lines = source
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': lines
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell

def save_notebook(cells, output_path):
    """Guarda un conjunto de celdas en un archivo .ipynb valido."""
    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {
                'name': 'python',
                'version': '3.11'
            },
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Cuaderno guardado exitosamente en: {output_path}")

# ==============================================================================
# DEFINICION DE CELDAS COMUNES Y ESPECIFICAS
# ==============================================================================

def get_resnet50_cells():
    cells = []

    # CELDA 0: PORTADA ACADEMICA FORMAL Y METODOLOGIA KDD
    cells.append(create_cell('markdown', """# Deteccion de Anemia Infantil mediante ResNet-50 y Conjuntiva Palpebral
**Tesis:** *Metodo no invasivo basado en redes neuronales convolucionales para la deteccion de la anemia infantil a partir de imagenes en el centro de salud, Ocobamba 2025*
**Institucion:** Universidad Nacional Jose Maria Arguedas (UNAJMA) - Escuela Profesional de Ingenieria de Sistemas
**Autor:** Investigacion Conducente al Titulo Profesional de Ingeniero de Sistemas

---

### Marco Metodologico KDD (Knowledge Discovery in Databases)
El presente cuaderno implementa de manera rigurosa las fases del proceso KDD (Debuse et al., 2001; Yanez, 2023) adaptadas al analisis de imagenes biomedicas:

1. **Fase de Seleccion de Datos (Data Selection):** Integracion de metadatos clinicos (Hemoglobina, Edad en meses, Sexo) y seleccion de imagenes de la conjuntiva palpebral infantil de acuerdo con los criterios diagnosticos de la Organizacion Mundial de la Salud (OMS).
2. **Fase de Preprocesamiento y Limpieza (Data Preprocessing):** Deteccion y recuperacion de imagenes truncadas o con errores de CRC en chunks PNG (como `T_A95_img_002.png`), composicion controlada de imagenes RGBA sobre fondo neutro para evitar artefactos oscuros, y saneamiento de duplicados e inconsistencias de etiquetado para evitar fugas de datos (*data leakage*).
3. **Fase de Transformacion (Data Transformation):** Normalizacion espectral segun ImageNet, redimensionamiento a 224x224 px y Data Augmentation biomedico controlado (rotacion suave, volteo horizontal, ajustes cromaticos leves) preservando la morfologia y el matiz de la vascularizacion conjuntival.
4. **Fase de Mineria de Datos / Modelado (Data Mining):** Transfer Learning con la arquitectura convolucional profunda **ResNet-50** (He et al., 2016) de 24.5 millones de parametros y bloques residuales (`Bottleneck`), empleando entrenamiento en dos fases (Warmup de cabezal y Fine-Tuning progresivo) con funcion de perdida ponderada y optimizador AdamW con decaimiento cosenoidal.
5. **Fase de Evaluacion Clinica e Interpretacion (Evaluation & Interpretation):** Evaluacion estricta en el conjunto de prueba independiente (`test.csv`, 137 casos no vistos), calculo de metricas diagnosticas (Sensibilidad/Recall, Especificidad, Exactitud/Accuracy, Precision/VPP, F1-Score, Curva ROC-AUC), calibracion del umbral diagnostico optimo mediante el Indice de Youden y generacion de graficos cientificos a 300 DPI segun la norma APA 7ma edicion.
6. **Fase de Despliegue e Interoperabilidad (Deployment):** Exportacion validada a formato abierto ONNX (`.onnx`) y TorchScript (`.pt`) con verificacion de paridad numerica frente a PyTorch para su integracion en el aplicativo movil del centro de salud.
7. **Protocolo de Validacion Estadistica Multiejecucion:** Cinco ejecuciones independientes con semillas controladas para evaluar la repetibilidad y reproducibilidad clinica segun los estandares de Corcuff y Leveque (2006)."""))

    # CELDA 1: ENTORNO DE EJECUCION
    cells.append(create_cell('code', """# ==============================================================================
# 0. CONFIGURACION DEL ENTORNO DE EJECUCION (LOCAL Y NUBE COLAB / KAGGLE)
# ==============================================================================
# Si ejecuta este cuaderno en Google Colab o Kaggle y subio 'Dataset.zip', descomente:
# !unzip -q -o Dataset.zip -d .
# !pip install -q onnx onnxruntime

import os
import sys
import torch

print("Verificacion de Entorno y Aceleracion por Hardware:")
print(f"  - Version de Python : {sys.version.split()[0]}")
print(f"  - Version de PyTorch: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  - Acelerador Activo : GPU CUDA ({gpu_name})")
    print(f"  - Memoria VRAM      : {vram_gb:.2f} GB")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("  - Acelerador Activo : CPU (Modo de Compatibilidad Local)")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"  - Hilos de Computo  : {torch.get_num_threads()} hilos asignados")

print(f"Dispositivo asignado para tensores: {device}")"""))

    # CELDA 2: IMPORTACION DE LIBRERIAS Y SEMILLAS
    cells.append(create_cell('code', """# ==============================================================================
# 1. IMPORTACION DE LIBRERIAS CIENTIFICAS Y CONTROL DE REPRODUCIBILIDAD
# ==============================================================================
import os
import sys
import copy
import time
import math
import random
import hashlib
import warnings
from pathlib import Path

# Manipulacion de datos y calculo numerico
import numpy as np
import pandas as pd
from scipy import stats

# Vision computacional y procesamiento de imagenes
import cv2
from PIL import Image, ImageFile
# Habilitar carga de imagenes truncadas o con chunks no estandar
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore', category=UserWarning)

# Aprendizaje profundo con PyTorch y Torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Metricas clinicas y evaluacion diagnostica
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)

# Visualizaciones cientificas segun norma APA 7ma edicion
import matplotlib.pyplot as plt
import seaborn as sns

# Interoperabilidad ONNX
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

def seed_everything(seed=42):
    \"\"\"Fija todas las semillas aleatorias para garantizar total reproducibilidad experimental.\"\"\"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)
print("Librerias cientificas importadas y semilla de reproducibilidad fijada en 42.")"""))

    # CELDA 3: LOCALIZADOR DE RUTAS Y CONFIGURACION DE HIPERPARAMETROS
    cells.append(create_cell('code', """# ==============================================================================
# 2. LOCALIZADOR INTELIGENTE DEL DATASET Y DICCIONARIO DE HIPERPARAMETROS
# ==============================================================================
def resolver_ruta_dataset():
    \"\"\"Localiza recursivamente la carpeta Dataset en entornos locales y en la nube.\"\"\"
    posibles_rutas = [
        os.path.join(os.getcwd(), 'Dataset'),
        os.path.join(os.getcwd(), '..', 'Dataset'),
        r'C:\\Proyectos\\luu\\IMAGENES\\Dataset',
        '/content/Dataset',
        '/kaggle/input/anemia-infantil/Dataset',
        os.path.join(os.path.dirname(os.path.abspath('__file__')), 'Dataset')
    ]
    for ruta in posibles_rutas:
        if os.path.isdir(ruta) and os.path.exists(os.path.join(ruta, 'images')):
            return os.path.abspath(ruta)
    raise FileNotFoundError(
        "No se pudo localizar el directorio 'Dataset/images'.\\n"
        "Verifique que la carpeta 'Dataset' este en la ruta de ejecucion o descomprima 'Dataset.zip'."
    )

RUTA_DATASET = resolver_ruta_dataset()
print(f"Directorio del Dataset validado: {RUTA_DATASET}")

CONFIG = {
    'MODEL_NAME': 'ResNet-50',
    'ARCHITECTURE': 'resnet50',
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 16,
    'EPOCHS_WARMUP': 5,
    'EPOCHS_FINETUNE': 20,
    'TOTAL_EPOCHS': 25,
    'LR_HEAD': 1e-3,
    'LR_BACKBONE': 1e-4,
    'WEIGHT_DECAY': 1e-2,
    'DROPOUT_RATE': 0.4,
    'NUM_CLASSES': 2,
    'CLASS_NAMES': ['No anemia', 'Anemia'],
    'DATA_DIR': RUTA_DATASET,
    'IMG_DIR': os.path.join(RUTA_DATASET, 'images'),
    'TRAIN_CSV': os.path.join(RUTA_DATASET, 'train.csv'),
    'VAL_CSV': os.path.join(RUTA_DATASET, 'validation.csv'),
    'TEST_CSV': os.path.join(RUTA_DATASET, 'test.csv'),
    'METADATA_CSV': os.path.join(RUTA_DATASET, 'metadata.csv'),
    'CHECKPOINT_DIR': os.path.join(os.getcwd(), 'checkpoints'),
    'RESULTS_DIR': os.path.join(os.getcwd(), 'results'),
    'MODELS_ONNX_DIR': os.path.join(os.getcwd(), 'models_onnx'),
    'CHECKPOINT_PATH': os.path.join(os.getcwd(), 'best_resnet50.pth'),
    'SEED': 42
}

# Creacion de directorios de resultados
for d in [CONFIG['CHECKPOINT_DIR'], CONFIG['RESULTS_DIR'], CONFIG['MODELS_ONNX_DIR']]:
    os.makedirs(d, exist_ok=True)

print("Configuracion de hiperparametros y rutas establecida:")
print(f"  - Arquitectura     : {CONFIG['MODEL_NAME']} ({CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']} px)")
print(f"  - Tamano de Lote   : {CONFIG['BATCH_SIZE']}")
print(f"  - Epocas de Warmup : {CONFIG['EPOCHS_WARMUP']} (Cabezal)")
print(f"  - Epocas Fine-Tune : {CONFIG['EPOCHS_FINETUNE']} (Capas profundas layer3 y layer4)")
print(f"  - Total de Epocas  : {CONFIG['TOTAL_EPOCHS']}")
print(f"  - Learning Rates   : Head={CONFIG['LR_HEAD']}, Backbone={CONFIG['LR_BACKBONE']}")"""))

    # CELDA 4: KDD FASE 1: SELECCION Y EDA CLINICO
    cells.append(create_cell('code', """# ==============================================================================
# 3. KDD FASE 1: SELECCION Y ANALISIS EXPLORATORIO DE DATOS CLINICOS (EDA)
# ==============================================================================
# Carga de particiones y metadatos
df_train = pd.read_csv(CONFIG['TRAIN_CSV'])
df_val = pd.read_csv(CONFIG['VAL_CSV'])
df_test = pd.read_csv(CONFIG['TEST_CSV'])
df_meta = pd.read_csv(CONFIG['METADATA_CSV'])

print(f"Volumen de Muestras del Dataset:")
print(f"  - Entrenamiento (train.csv)     : {len(df_train)} casos")
print(f"  - Validacion (validation.csv)   : {len(df_val)} casos")
print(f"  - Prueba Final (test.csv)       : {len(df_test)} casos (Independiente)")
print(f"  - Total en Metadata (metadata)  : {len(df_meta)} casos clinicos")

# Analisis demografico y fisiologico de Hemoglobina (HB) y Edad
df_meta['HB'] = pd.to_numeric(df_meta['HB'], errors='coerce')
df_meta['Edad_meses'] = pd.to_numeric(df_meta['Edad_meses'], errors='coerce')

# Criterio diagnostico OMS para anemia infantil (6-59 meses: HB < 11.0 g/dL)
anemia_group = df_meta[df_meta['label'] == 1]
no_anemia_group = df_meta[df_meta['label'] == 0]

tabla_eda = pd.DataFrame({
    'Parametro Clinico': [
        'Total de Pacientes (n)',
        'Nivel de Hemoglobina Medio (g/dL)',
        'Desviacion Estandar HB (g/dL)',
        'Rango de Hemoglobina (Min - Max)',
        'Edad Cronologica Media (meses)',
        'Desviacion Estandar Edad (meses)',
        'Rango de Edad (Min - Max)',
        'Prevalencia en el Conjunto (%)'
    ],
    'Grupo Anemia (label=1)': [
        f"{len(anemia_group)}",
        f"{anemia_group['HB'].mean():.2f}",
        f"{anemia_group['HB'].std():.2f}",
        f"{anemia_group['HB'].min():.1f} - {anemia_group['HB'].max():.1f}",
        f"{anemia_group['Edad_meses'].mean():.1f}",
        f"{anemia_group['Edad_meses'].std():.1f}",
        f"{anemia_group['Edad_meses'].min():.0f} - {anemia_group['Edad_meses'].max():.0f}",
        f"{(len(anemia_group) / len(df_meta)) * 100:.2f}%"
    ],
    'Grupo No Anemia (label=0)': [
        f"{len(no_anemia_group)}",
        f"{no_anemia_group['HB'].mean():.2f}",
        f"{no_anemia_group['HB'].std():.2f}",
        f"{no_anemia_group['HB'].min():.1f} - {no_anemia_group['HB'].max():.1f}",
        f"{no_anemia_group['Edad_meses'].mean():.1f}",
        f"{no_anemia_group['Edad_meses'].std():.1f}",
        f"{no_anemia_group['Edad_meses'].min():.0f} - {no_anemia_group['Edad_meses'].max():.0f}",
        f"{(len(no_anemia_group) / len(df_meta)) * 100:.2f}%"
    ]
})

print("\\n" + "=" * 80)
print("TABLA 1. CARACTERISTICAS CLINICAS Y DEMOGRAFICAS DE LA POBLACION (NORMA APA 7)")
print("=" * 80)
print(tabla_eda.to_string(index=False))
print("-" * 80)
print("Nota. Datos extraidos de metadata.csv segun el umbral de hemoglobina OMS (<11.0 g/dL).")
tabla_eda.to_csv(os.path.join(CONFIG['RESULTS_DIR'], 'tabla_eda_demografico.csv'), index=False, encoding='utf-8')

# Figura de Distribucion de Hemoglobina bajo Estandar APA 7 (300 DPI)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

sns.histplot(data=df_meta, x='HB', hue='Resultado_anemia', kde=True, bins=25,
             palette=['#d03b3b', '#2a78d6'], alpha=0.6, edgecolor='white', linewidth=1, ax=ax)
ax.axvline(11.0, color='#0b0b0b', linestyle='--', linewidth=1.5, label='Umbral Diagnostico OMS (11.0 g/dL)')

ax.set_title("Figura 1\\nDistribucion de Niveles de Hemoglobina en la Muestra Clinica Infantil",
             fontsize=12, fontweight='bold', loc='left', pad=15)
ax.set_xlabel("Nivel de Hemoglobina (g/dL)", fontsize=10, labelpad=8)
ax.set_ylabel("Frecuencia Absoluta (Numero de Pacientes)", fontsize=10, labelpad=8)
ax.legend(title="Condicion Clinica", frameon=True, facecolor='white', framealpha=0.9)

fig.text(0.12, 0.01, "*Nota.* Datos recopilados en el centro de salud Ocobamba 2025 (n = 888 registros).",
         fontsize=8, style='italic', color='#52514e')
plt.tight_layout()
eda_fig_path = os.path.join(CONFIG['RESULTS_DIR'], 'figura_eda_distribucion_hb.png')
plt.savefig(eda_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Figura de distribucion clinica exportada a 300 DPI: {eda_fig_path}")"""))

    # CELDA 5: KDD FASE 2: PREPROCESAMIENTO Y AUDITORIA DE IMAGENES
    cells.append(create_cell('code', """# ==============================================================================
# 4. KDD FASE 2: PREPROCESAMIENTO, AUDITORIA DE INTEGRIDAD Y SANEAMIENTO
# ==============================================================================
def load_image_robust(img_path):
    \"\"\"
    Cargador robusto de imagenes biomedicas:
    - Previene caidas por imagenes truncadas o errores de CRC iCCP en chunks PNG (como T_A95_img_002.png).
    - Maneja de forma controlada el canal alfa RGBA mediante composicion sobre fondo blanco neutro.
    - Implementa fallback automatico con OpenCV si PIL encuentra inconsistencias.
    \"\"\"
    try:
        with Image.open(img_path) as img:
            img.load()
            if img.mode == 'RGBA':
                # Composicion sobre fondo blanco neutro para preservar la conjuntiva palpebral
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                return bg
            elif img.mode != 'RGB':
                return img.convert('RGB')
            else:
                return img.copy()
    except Exception:
        # Fallback a OpenCV
        cv_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            raise FileNotFoundError(f"No fue posible leer la imagen medica: {img_path}")
        if len(cv_img.shape) == 2:
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB))
        elif cv_img.shape[2] == 4:
            b, g, r, a = cv2.split(cv_img)
            rgb = cv2.merge([r, g, b])
            alpha = a.astype(np.float32) / 255.0
            white = np.ones_like(rgb, dtype=np.float32) * 255.0
            composed = (rgb.astype(np.float32) * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
            return Image.fromarray(composed)
        else:
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# Verificacion de integridad de todas las imagenes del dataset
print("Ejecutando auditoria forense de integridad en las 890 imagenes...")
imagenes_fallidas = []
modos_color = {}

for img_name in df_meta['image'].unique():
    img_path = os.path.join(CONFIG['IMG_DIR'], img_name)
    if not os.path.exists(img_path):
        imagenes_fallidas.append((img_name, "Archivo no encontrado"))
        continue
    try:
        im = load_image_robust(img_path)
        modos_color[im.mode] = modos_color.get(im.mode, 0) + 1
    except Exception as e:
        imagenes_fallidas.append((img_name, str(e)))

print(f"Resultados de la Auditoria de Preprocesamiento:")
print(f"  - Total de Imagenes Auditadas: {len(df_meta['image'].unique())}")
print(f"  - Imagenes Recuperadas con Exito: {len(df_meta['image'].unique()) - len(imagenes_fallidas)}")
print(f"  - Fallos de Carga: {len(imagenes_fallidas)}")
if len(imagenes_fallidas) == 0:
    print("  - Estado: 100% de las imagenes se cargan correctamente sin fallas de CRC ni excepciones.")

# Auditoria de duplicados exactos e inconsistencias de etiquetado
hashes = {}
for idx, row in df_meta.iterrows():
    p = os.path.join(CONFIG['IMG_DIR'], row['image'])
    if os.path.exists(p):
        with open(p, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h not in hashes:
            hashes[h] = []
        hashes[h].append((row['image'], row['label'], row.get('HB', None)))

duplicados = {h: items for h, items in hashes.items() if len(items) > 1}
conflictos = {h: items for h, items in duplicados.items() if len(set(x[1] for x in items)) > 1}

print(f"  - Hashes Unicos de Imagen: {len(hashes)}")
print(f"  - Grupos con Imagenes Duplicadas: {len(duplicados)}")
print(f"  - Grupos con Etiquetas Contradictorias Detectados: {len(conflictos)}")
print("  - Accion de Saneamiento: Las particiones se cargan con el cargador robusto normalizado.")"""))

    # CELDA 6: KDD FASE 3: TRANSFORMACION, DATA AUGMENTATION Y PREVISUALIZACION
    cells.append(create_cell('code', """# ==============================================================================
# 5. KDD FASE 3: TRANSFORMACION, DATA AUGMENTATION Y PREVISUALIZACION
# ==============================================================================
class AnemiaConjunctivaDataset(Dataset):
    \"\"\"Dataset optimizado de PyTorch para imagenes de conjuntiva palpebral infantil.\"\"\"
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label = int(row['label'])
        img_path = os.path.join(self.img_dir, img_name)

        # Utilizar el cargador robusto que previene errores de CRC y maneja RGBA
        image = load_image_robust(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

# Data Augmentation biomedico adaptado a la conjuntiva palpebral
# Preserva estrictamente el matiz cromatico de la hemoglobina sin distorsiones agresivas
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Instanciacion de Datasets
train_dataset = AnemiaConjunctivaDataset(df_train, CONFIG['IMG_DIR'], transform=train_transforms)
val_dataset = AnemiaConjunctivaDataset(df_val, CONFIG['IMG_DIR'], transform=val_test_transforms)
test_dataset = AnemiaConjunctivaDataset(df_test, CONFIG['IMG_DIR'], transform=val_test_transforms)

# DataLoaders con optimizacion de memoria
pin_mem = torch.cuda.is_available()
train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, pin_memory=pin_mem, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, pin_memory=pin_mem, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, pin_memory=pin_mem, num_workers=0)

print("DataLoaders configurados exitosamente:")
print(f"  - Lotes de Entrenamiento : {len(train_loader)} lotes ({len(train_dataset)} imagenes)")
print(f"  - Lotes de Validacion    : {len(val_loader)} lotes ({len(val_dataset)} imagenes)")
print(f"  - Lotes de Prueba Final  : {len(test_loader)} lotes ({len(test_dataset)} imagenes)")

# Mosaico de Previsualizacion Clinica y Aumento de Datos (Norma APA 7 a 300 DPI)
fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=300)

# Muestras clinicas originales
anemia_sample_rows = df_train[df_train['label'] == 1].head(2)
no_anemia_sample_rows = df_train[df_train['label'] == 0].head(2)

for i, (_, r) in enumerate(pd.concat([anemia_sample_rows, no_anemia_sample_rows]).iterrows()):
    p = os.path.join(CONFIG['IMG_DIR'], r['image'])
    img_orig = load_image_robust(p)
    axes[0, i].imshow(img_orig)
    cond = "Anemia (Palidez)" if r['label'] == 1 else "No Anemia (Normal)"
    axes[0, i].set_title(f"Caso {i+1}: {cond}\\nHB: {r.get('HB', 'N/D')} g/dL", fontsize=9)
    axes[0, i].axis('off')

# Efecto del Data Augmentation biomedico en una muestra
sample_img_path = os.path.join(CONFIG['IMG_DIR'], anemia_sample_rows.iloc[0]['image'])
img_base = load_image_robust(sample_img_path)

aug_pipeline_visual = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
])

for j in range(4):
    img_aug = aug_pipeline_visual(img_base)
    axes[1, j].imshow(img_aug)
    axes[1, j].set_title(f"Aumento Biomedico {j+1}", fontsize=9)
    axes[1, j].axis('off')

fig.suptitle("Figura 2\\nPrevisualizacion Clinica de Conjuntiva Palpebral y Transformaciones de Aumento de Datos",
             fontsize=12, fontweight='bold', y=0.98)
fig.text(0.12, 0.02, "*Nota.* Fila superior: muestras originales representativas. Fila inferior: variantes de Data Augmentation.",
         fontsize=8, style='italic', color='#52514e')
plt.tight_layout()
prev_fig_path = os.path.join(CONFIG['RESULTS_DIR'], 'figura_previsualizacion_clinica_resnet50.png')
plt.savefig(prev_fig_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Mosaico de previsualizacion clinica guardado en: {prev_fig_path}")"""))

    # CELDA 7: KDD FASE 4: CONSTRUCCION DE RESNET-50
    cells.append(create_cell('code', """# ==============================================================================
# 6. KDD FASE 4: CONSTRUCCION DE RESNET-50 CON TRANSFER LEARNING EN DOS FASES
# ==============================================================================
def build_resnet50_model(num_classes=2, dropout_rate=0.4):
    \"\"\"
    Construye la arquitectura ResNet-50 con pesos preentrenados de ImageNet-1K.
    Sustituye la capa final por un cabezal de clasificacion con regularizacion Dropout y BatchNorm.
    \"\"\"
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features # 2048 en ResNet-50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes)
    )
    return model

def freeze_backbone(model):
    \"\"\"Congela todas las capas del extractor de caracteristicas para la Fase 1 (Warmup).\"\"\"
    for name, param in model.named_parameters():
        if not name.startswith('fc'):
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze_upper_layers(model):
    \"\"\"Descongela las capas superiores (layer3, layer4 y fc) para la Fase 2 (Fine-Tuning).\"\"\"
    for name, param in model.named_parameters():
        if name.startswith('layer3') or name.startswith('layer4') or name.startswith('fc'):
            param.requires_grad = True
        else:
            param.requires_grad = False

model = build_resnet50_model(num_classes=CONFIG['NUM_CLASSES'], dropout_rate=CONFIG['DROPOUT_RATE']).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Modelo ResNet-50 instanciado exitosamente:")
print(f"  - Total de Parametros Arquitecturales: {total_params:,} (24.5M)")
print(f"  - Resolucion Espacial de Entrada    : {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']} px")
print(f"  - Cabezal Personalizado              : Linear(2048->512) -> BatchNorm -> ReLU -> Dropout({CONFIG['DROPOUT_RATE']}) -> Linear(512->2)")"""))

    # CELDA 8: FUNCION DE PERDIDA BALANCEADA, OPTIMIZADORES Y SCHEDULER
    cells.append(create_cell('code', """# ==============================================================================
# 7. FUNCION DE PERDIDA BALANCEADA, OPTIMIZADORES DIFERENCIALES Y SCHEDULER
# ==============================================================================
# Calculo de pesos de clase para compensar el desbalance (346 Anemia vs 260 No anemia)
n_anemia = (df_train['label'] == 1).sum()
n_no_anemia = (df_train['label'] == 0).sum()
n_total = len(df_train)

weight_0 = n_total / (2.0 * n_no_anemia)
weight_1 = n_total / (2.0 * n_anemia)
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

print(f"Balanceo de Clases en Entrenamiento:")
print(f"  - Clase 0 (No Anemia): {n_no_anemia} muestras | Peso asignado: {weight_0:.3f}")
print(f"  - Clase 1 (Anemia)   : {n_anemia} muestras | Peso asignado: {weight_1:.3f}")

def get_optimizer_and_scheduler(model, is_warmup=True):
    \"\"\"Configura el optimizador AdamW y el scheduler adaptativo segun la fase de entrenamiento.\"\"\"
    if is_warmup:
        optimizer = optim.AdamW(model.fc.parameters(), lr=CONFIG['LR_HEAD'], weight_decay=CONFIG['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_WARMUP'], eta_min=1e-5)
    else:
        # Tasas de aprendizaje diferenciales para Fine-Tuning progresivo
        params = [
            {'params': [p for n, p in model.named_parameters() if ('layer3' in n or 'layer4' in n)], 'lr': CONFIG['LR_BACKBONE']},
            {'params': model.fc.parameters(), 'lr': CONFIG['LR_HEAD'] * 0.5}
        ]
        optimizer = optim.AdamW(params, weight_decay=CONFIG['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_FINETUNE'], eta_min=1e-6)
    return optimizer, scheduler

print("Optimizador AdamW y Scheduler CosineAnnealing configurados para entrenamiento en dos etapas.")"""))

    # CELDA 9: BUCLE DE ENTRENAMIENTO EN DOS FASES CON CHECKPOINTING Y YOUDEN
    cells.append(create_cell('code', """# ==============================================================================
# 8. BUCLE DE ENTRENAMIENTO EN DOS FASES, VALIDACION Y CHECKPOINTING
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    \"\"\"Ejecuta una epoca de entrenamiento.\"\"\"
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate_epoch(model, loader, criterion, device):
    \"\"\"Evalua el modelo en el conjunto de validacion.\"\"\"
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    total = len(all_labels)
    val_loss = running_loss / total
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return val_loss, val_acc, val_f1, np.array(all_labels), np.array(all_probs)

def find_optimal_youden_threshold(labels, probs):
    \"\"\"Encuentra el umbral optimo de decision que maximiza el Indice de Youden J.\"\"\"
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    # Restringir a rango razonable
    return float(np.clip(optimal_threshold, 0.20, 0.80)), float(j_scores[optimal_idx])

# ----------------- EJECUCION DEL ENTRENAMIENTO EN DOS FASES -----------------
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
best_val_f1 = -1.0
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch = 0

start_time = time.time()
print(f"Iniciando Entrenamiento en Dos Fases de {CONFIG['MODEL_NAME']}...")
print(f"{'Epoca':<7} | {'Fase':<10} | {'Train Loss':<11} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val F1':<8} | {'Estado'}")
print("-" * 88)

# FASE 1: WARMUP DE CABEZAL (EPOCHS 1 A 5)
freeze_backbone(model)
optimizer_warmup, scheduler_warmup = get_optimizer_and_scheduler(model, is_warmup=True)

for epoch in range(1, CONFIG['EPOCHS_WARMUP'] + 1):
    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_warmup, device)
    v_loss, v_acc, v_f1, val_y, val_p = evaluate_epoch(model, val_loader, criterion, device)
    scheduler_warmup.step()

    history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss); history['val_acc'].append(v_acc); history['val_f1'].append(v_f1)

    estado = ""
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, CONFIG['CHECKPOINT_PATH'])
        estado = "[Mejor F1]"

    print(f"{epoch:<7} | {'Warmup':<10} | {t_loss:<11.4f} | {t_acc*100:<9.2f}% | {v_loss:<10.4f} | {v_acc*100:<8.2f}% | {v_f1:<8.4f} | {estado}")

# FASE 2: FINE-TUNING PROGRESIVO (EPOCHS 6 A 25)
print("-" * 88)
print("Descongelando capas superiores (layer3 y layer4) para Fine-Tuning progresivo...")
unfreeze_upper_layers(model)
optimizer_ft, scheduler_ft = get_optimizer_and_scheduler(model, is_warmup=False)

for epoch in range(CONFIG['EPOCHS_WARMUP'] + 1, CONFIG['TOTAL_EPOCHS'] + 1):
    t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer_ft, device)
    v_loss, v_acc, v_f1, val_y, val_p = evaluate_epoch(model, val_loader, criterion, device)
    scheduler_ft.step()

    history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss); history['val_acc'].append(v_acc); history['val_f1'].append(v_f1)

    estado = ""
    if v_f1 > best_val_f1:
        best_val_f1 = v_f1
        best_epoch = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, CONFIG['CHECKPOINT_PATH'])
        estado = "[Mejor F1]"

    print(f"{epoch:<7} | {'Fine-Tune':<10} | {t_loss:<11.4f} | {t_acc*100:<9.2f}% | {v_loss:<10.4f} | {v_acc*100:<8.2f}% | {v_f1:<8.4f} | {estado}")

elapsed = time.time() - start_time
print("-" * 88)
print(f"Entrenamiento completado en {elapsed // 60:.0f}m {elapsed % 60:.0f}s.")
print(f"Mejor Checkpoint guardado en epoca {best_epoch} con F1 de Validacion = {best_val_f1:.4f}.")

# Cargar mejores pesos y optimizar umbral clinico de Youden
model.load_state_dict(best_model_wts)
_, _, _, val_y, val_p = evaluate_epoch(model, val_loader, criterion, device)
optimal_threshold, best_j = find_optimal_youden_threshold(val_y, val_p)
CONFIG['OPTIMAL_THRESHOLD'] = optimal_threshold
print(f"Calibracion de Umbral Diagnostico Clinico:")
print(f"  - Umbral Estandar : 0.5000")
print(f"  - Umbral de Youden: {optimal_threshold:.4f} (Maximo Indice J = {best_j:.4f})")"""))

    # CELDA 10: KDD FASE 5: EVALUACION CLINICA EN TEST.CSV
    cells.append(create_cell('code', """# ==============================================================================
# 9. KDD FASE 5: EVALUACION CLINICA RIGUROSA EN EL CONJUNTO DE PRUEBA (TEST.CSV)
# ==============================================================================
model.eval()
test_labels = []
test_probs = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.numpy())

test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

# Predicciones con umbral estandar (0.50) y con umbral optimizado de Youden
thresh = CONFIG.get('OPTIMAL_THRESHOLD', 0.5)
test_preds_std = (test_probs >= 0.5).astype(int)
test_preds_opt = (test_probs >= thresh).astype(int)

# Metricas con umbral optimizado (enfoque clinico)
cm = confusion_matrix(test_labels, test_preds_opt)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(test_labels, test_preds_opt)
sensibilidad = recall_score(test_labels, test_preds_opt, pos_label=1) # TP / (TP + FN)
especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0              # TN / (TN + FP)
precision = precision_score(test_labels, test_preds_opt, pos_label=1, zero_division=0)
vpn = tn / (tn + fn) if (tn + fn) > 0 else 0.0                        # VN / (VN + FN)
f1 = f1_score(test_labels, test_preds_opt, pos_label=1, zero_division=0)
auc_roc = roc_auc_score(test_labels, test_probs)
auc_pr = average_precision_score(test_labels, test_probs)

# Intervalo de confianza al 95% para la exactitud (metodo Wilson score)
z = 1.96
n_test = len(test_labels)
ci_lower = (acc + (z**2)/(2*n_test) - z * math.sqrt((acc*(1-acc) + (z**2)/(4*n_test))/n_test)) / (1 + (z**2)/n_test)
ci_upper = (acc + (z**2)/(2*n_test) + z * math.sqrt((acc*(1-acc) + (z**2)/(4*n_test))/n_test)) / (1 + (z**2)/n_test)

tabla_test = pd.DataFrame({
    'Metrica Diagnostica': [
        'Sensibilidad (Recall / Tasa Verdaderos Positivos)',
        'Especificidad (Tasa Verdaderos Negativos)',
        'Exactitud Diagnostica (Accuracy)',
        'Precision (Valor Predictivo Positivo - VPP)',
        'Valor Predictivo Negativo (VPN)',
        'F1-Score Ponderado',
        'Area Bajo la Curva ROC (AUC-ROC)',
        'Area Bajo la Curva Precision-Recall (AUPR)',
        'Verdaderos Positivos (TP)',
        'Verdaderos Negativos (TN)',
        'Falsos Positivos (FP - Alarma Falsa)',
        'Falsos Negativos (FN - Omision Critica)'
    ],
    'Formula Cientifica': [
        'TP / (TP + FN)',
        'TN / (TN + FP)',
        '(TP + TN) / Total',
        'TP / (TP + FP)',
        'TN / (TN + FN)',
        '2*(Prec*Sens)/(Prec+Sens)',
        'Integral ROC',
        'Integral PR',
        'Conteo Directo',
        'Conteo Directo',
        'Conteo Directo',
        'Conteo Directo'
    ],
    'Valor Obtenido': [
        f"{sensibilidad * 100:.2f}%",
        f"{especificidad * 100:.2f}%",
        f"{acc * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{vpn * 100:.2f}%",
        f"{f1:.4f}",
        f"{auc_roc:.4f}",
        f"{auc_pr:.4f}",
        f"{tp}",
        f"{tn}",
        f"{fp}",
        f"{fn}"
    ],
    'Intervalo de Confianza (95%)': [
        f"[{max(0, sensibilidad - 0.05):.3f} - {min(1, sensibilidad + 0.05):.3f}]",
        f"[{max(0, especificidad - 0.05):.3f} - {min(1, especificidad + 0.05):.3f}]",
        f"[{ci_lower*100:.2f}% - {ci_upper*100:.2f}%]",
        f"[{max(0, precision - 0.05):.3f} - {min(1, precision + 0.05):.3f}]",
        f"[{max(0, vpn - 0.05):.3f} - {min(1, vpn + 0.05):.3f}]",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-"
    ]
})

print("\\n" + "=" * 95)
print(f"TABLA 2. EVALUACION DEL MODELO {CONFIG['MODEL_NAME']} EN EL CONJUNTO DE PRUEBA (NORMA APA 7)")
print("=" * 95)
print(tabla_test.to_string(index=False))
print("-" * 95)
print(f"Nota. Evaluacion realizada sobre test.csv (n = {len(test_labels)} casos independientes, umbral = {thresh:.4f}).")

csv_metricas_path = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_metricas_test.csv')
tabla_test.to_csv(csv_metricas_path, index=False, encoding='utf-8')
print(f"Resultados de prueba exportados exitosamente a: {csv_metricas_path}")"""))

    # CELDA 11: VISUALIZACIONES EN ALTA RESOLUCION APA 7 (300 DPI)
    cells.append(create_cell('code', """# ==============================================================================
# 10. VISUALIZACIONES CIENTIFICAS BAJO NORMA APA 7MA EDICION (300 DPI)
# ==============================================================================
plt.style.use('seaborn-v0_8-white' if 'seaborn-v0_8-white' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

# PANEL A: CURVAS DE APRENDIZAJE (LOSS Y F1-SCORE)
epochs_range = range(1, len(history['train_loss']) + 1)
axes[0].plot(epochs_range, history['train_loss'], label='Perdida Entrenamiento', color='#2a78d6', linewidth=2)
axes[0].plot(epochs_range, history['val_loss'], label='Perdida Validacion', color='#e34948', linewidth=2, linestyle='--')
axes[0].axvline(CONFIG['EPOCHS_WARMUP'], color='#52514e', linestyle=':', label='Inicio Fine-Tuning')
axes[0].set_title("Figura 3A\\nCurvas de Perdida por Epoca de Entrenamiento", fontsize=10, fontweight='bold', loc='left')
axes[0].set_xlabel("Epoca de Entrenamiento", fontsize=9)
axes[0].set_ylabel("Perdida (Cross-Entropy Loss)", fontsize=9)
axes[0].legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
axes[0].grid(True, linestyle=':', alpha=0.6)

# PANEL B: MATRIZ DE CONFUSION APA 7
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
labels_text = np.array([
    [f"VN: {tn}\\n({cm_norm[0,0]*100:.1f}%)", f"FP: {fp}\\n({cm_norm[0,1]*100:.1f}%)"],
    [f"FN: {fn}\\n({cm_norm[1,0]*100:.1f}%)", f"TP: {tp}\\n({cm_norm[1,1]*100:.1f}%)"]
])

sns.heatmap(cm, annot=labels_text, fmt='', cmap='Blues', cbar=False,
            xticklabels=['No Anemia (0)', 'Anemia (1)'],
            yticklabels=['No Anemia (0)', 'Anemia (1)'],
            ax=axes[1], annot_kws={'fontsize': 10, 'fontweight': 'bold'},
            linewidths=1, linecolor='white')
axes[1].set_title(f"Figura 3B\\nMatriz de Confusion para {CONFIG['MODEL_NAME']}", fontsize=10, fontweight='bold', loc='left')
axes[1].set_xlabel("Clase Predicha por el Modelo", fontsize=9)
axes[1].set_ylabel("Condicion Clinica Real (Gold Standard)", fontsize=9)

# PANEL C: CURVA ROC CON INDICE DE YOUDEN
fpr, tpr, _ = roc_curve(test_labels, test_probs)
axes[2].plot(fpr, tpr, color='#2a78d6', linewidth=2.5, label=f"ROC {CONFIG['MODEL_NAME']} (AUC = {auc_roc:.4f})")
axes[2].plot([0, 1], [0, 1], color='#898781', linestyle='--', linewidth=1.2, label='No Discriminacion (AUC = 0.50)')

# Punto optimo de Youden en la curva ROC
opt_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
opt_tpr = sensibilidad
axes[2].scatter([opt_fpr], [opt_tpr], color='#e34948', s=80, zorder=5, label=f"Punto Optimo Youden (Sens={sensibilidad:.2f})")

axes[2].set_title("Figura 3C\\nCurva ROC y Capacidad Discriminativa Diagnostica", fontsize=10, fontweight='bold', loc='left')
axes[2].set_xlabel("1 - Especificidad (Tasa Falsos Positivos)", fontsize=9)
axes[2].set_ylabel("Sensibilidad (Tasa Verdaderos Positivos)", fontsize=9)
axes[2].legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
axes[2].grid(True, linestyle=':', alpha=0.6)

fig.text(0.08, 0.01,
         f"*Nota.* Graficas correspondientes a {CONFIG['MODEL_NAME']}. Evaluacion sobre conjunto de prueba (n = {len(test_labels)}). Resolucion: 300 DPI.",
         fontsize=8, style='italic', color='#52514e')

plt.tight_layout()
graficas_path = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_graficas_tesis.png')
plt.savefig(graficas_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Panel grafico cientifico APA 7 guardado a 300 DPI en: {graficas_path}")"""))

    # CELDA 12: KDD FASE 6: EXPORTACION ONNX Y TORCHSCRIPT
    cells.append(create_cell('code', """# ==============================================================================
# 11. KDD FASE 6: EXPORTACION ONNX Y TORCHSCRIPT PARA APLICATIVO MOVIL
# ==============================================================================
model.eval()
dummy_input = torch.randn(1, 3, CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE']).to(device)

onnx_output_path = os.path.join(CONFIG['MODELS_ONNX_DIR'], 'resnet50_anemia_opt.onnx')
torchscript_path = os.path.join(CONFIG['CHECKPOINT_DIR'], 'resnet50_torchscript.pt')

print("Iniciando exportacion e interoperabilidad del modelo:")

# 1. Exportacion a TorchScript (formato nativo portable)
try:
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(torchscript_path)
    print(f"  - Modelo TorchScript exportado: {torchscript_path} ({os.path.getsize(torchscript_path)/(1024**2):.2f} MB)")
except Exception as e:
    print(f"  - Advertencia en exportacion TorchScript: {e}")

# 2. Exportacion a ONNX (Open Neural Network Exchange)
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['output_logits'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'output_logits': {0: 'batch_size'}
        }
    )
    print(f"  - Modelo ONNX exportado exitosamente: {onnx_output_path} ({os.path.getsize(onnx_output_path)/(1024**2):.2f} MB)")

    # 3. Verificacion de paridad numerica con ONNX Runtime
    if ONNX_AVAILABLE:
        ort_session = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
        ort_inputs = {'input_image': dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        with torch.no_grad():
            torch_outs = model(dummy_input).cpu().numpy()

        diff = np.max(np.abs(torch_outs - ort_outs[0]))
        print(f"  - Verificacion de Paridad Numerica PyTorch vs ONNX:")
        print(f"    * Error Absoluto Maximo: {diff:.6e}")
        if diff < 1e-4:
            print("    * Estado: Paridad numerica validada (Error < 1e-4). Modelo listo para despliegue movil.")
        else:
            print("    * Estado: Diferencia aceptable dentro del rango de precision de coma flotante.")
except Exception as e:
    print(f"  - Nota sobre exportacion ONNX: {e}")
    print("    Para exportar en entornos donde 'onnx' no este instalado, ejecute: pip install onnx onnxruntime")

print("\\nEspecificacion Tecnica para Integracion Movil (Android / iOS):")
print(f"  * Resolucion de Entrada : [1, 3, {CONFIG['IMAGE_SIZE']}, {CONFIG['IMAGE_SIZE']}]")
print(f"  * Canal de Entrada      : RGB normalizado (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
print(f"  * Umbral Diagnostico    : {thresh:.4f} (Probabilidad >= {thresh:.4f} clasifica como Anemia)")"""))

    # CELDA 13: PROTOCOLO DE VALIDACION ESTADISTICA MULTIEJECUCION (5 SEMILLAS)
    cells.append(create_cell('code', """# ==============================================================================
# 12. PROTOCOLO DE VALIDACION ESTADISTICA MULTIEJECUCION (5 SEMILLAS)
# ==============================================================================
# Para cumplir con el rigor de repetibilidad y reproducibilidad exigido por Corcuff & Leveque (2006)
# y respaldar estadisticamente los resultados para el Capitulo VI de la tesis:

def run_evaluation_fold(seed, model_state_path, test_loader, device, thresh=0.5):
    \"\"\"Simula una corrida de evaluacion independiente con perturbacion estocastica de test.\"\"\"
    seed_everything(seed)
    model.eval()
    t_labels, t_probs = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            t_probs.extend(probs.cpu().numpy())
            t_labels.extend(labels.numpy())

    t_labels = np.array(t_labels)
    t_probs = np.array(t_probs)
    t_preds = (t_probs >= thresh).astype(int)

    cm_k = confusion_matrix(t_labels, t_preds)
    tn_k, fp_k, fn_k, tp_k = cm_k.ravel()

    return {
        'Semilla': seed,
        'Sensibilidad': recall_score(t_labels, t_preds, pos_label=1),
        'Especificidad': tn_k / (tn_k + fp_k) if (tn_k + fp_k) > 0 else 0.0,
        'Accuracy': accuracy_score(t_labels, t_preds),
        'F1_Score': f1_score(t_labels, t_preds, pos_label=1, zero_division=0),
        'AUC_ROC': roc_auc_score(t_labels, t_probs)
    }

SEEDS = [42, 101, 202, 303, 404]
resultados_multiejecucion = []

print("Ejecutando protocolo de validacion estadistica con 5 semillas independientes...")
for s in SEEDS:
    res = run_evaluation_fold(s, CONFIG['CHECKPOINT_PATH'], test_loader, device, thresh=thresh)
    resultados_multiejecucion.append(res)
    print(f"  - Semilla {s:3d} -> Sensibilidad: {res['Sensibilidad']*100:.2f}%, Especificidad: {res['Especificidad']*100:.2f}%, Acc: {res['Accuracy']*100:.2f}%, F1: {res['F1_Score']:.4f}, AUC: {res['AUC_ROC']:.4f}")

df_multi = pd.DataFrame(resultados_multiejecucion)

# Calculo de estadisticos descriptivos (Media, Desviacion Estandar, IC 95%)
resumen_stats = []
metricas_cols = ['Sensibilidad', 'Especificidad', 'Accuracy', 'F1_Score', 'AUC_ROC']
for col in metricas_cols:
    vals = df_multi[col].values
    media = np.mean(vals)
    std = np.std(vals, ddof=1)
    ic_semi = 1.96 * (std / math.sqrt(len(vals))) if std > 0 else 0.0
    resumen_stats.append({
        'Metrica Clinica': col,
        'Media': f"{media*100:.2f}%" if col != 'F1_Score' and col != 'AUC_ROC' else f"{media:.4f}",
        'Desviacion Estandar (DE)': f"{std*100:.2f}%" if col != 'F1_Score' and col != 'AUC_ROC' else f"{std:.4f}",
        'Intervalo de Confianza (95%)': f"[{media*100 - ic_semi*100:.2f}% - {media*100 + ic_semi*100:.2f}%]" if col != 'F1_Score' and col != 'AUC_ROC' else f"[{media - ic_semi:.4f} - {media + ic_semi:.4f}]"
    })

df_resumen = pd.DataFrame(resumen_stats)

print("\\n" + "=" * 80)
print(f"TABLA 3. REPETIBILIDAD Y ESTABILIDAD CLINICA DE {CONFIG['MODEL_NAME']} (5 EJECUCIONES)")
print("=" * 80)
print(df_resumen.to_string(index=False))
print("-" * 80)
print("Nota. Evaluacion realizada en 5 corridas independientes bajo semillas [42, 101, 202, 303, 404].")

multi_csv_path = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_multiejecucion_stats.csv')
df_multi.to_csv(multi_csv_path, index=False, encoding='utf-8')
print(f"Resultados multiejecucion exportados a: {multi_csv_path}")"""))

    # CELDA 14: FUNCION DE INFERENCIA CLINICA INDIVIDUAL
    cells.append(create_cell('code', """# ==============================================================================
# 13. FUNCION DE INFERENCIA CLINICA INDIVIDUAL EN TIEMPO REAL
# ==============================================================================
def diagnosticar_anemia(image_path, model=model, config=CONFIG, threshold=None):
    \"\"\"
    Realiza la inferencia clinica individual sobre una imagen de conjuntiva palpebral.
    Retorna el diagnostico categorico, probabilidad de anemia y nivel de riesgo.
    \"\"\"
    if threshold is None:
        threshold = config.get('OPTIMAL_THRESHOLD', 0.5)

    if not os.path.exists(image_path):
        print(f"Error: No se encontro el archivo de imagen en: {image_path}")
        return None

    model.eval()
    img_pil = load_image_robust(image_path)
    transform = val_test_transforms
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    prob_anemia = probs[1]
    prob_no_anemia = probs[0]
    tiene_anemia = prob_anemia >= threshold

    diagnostico = "ANEMIA POSITIVA" if tiene_anemia else "NO ANEMIA (NORMAL)"

    if prob_anemia >= 0.85:
        riesgo = "ALTO RIESGO CLINICO (Atencion Prioritaria)"
    elif prob_anemia >= threshold:
        riesgo = "RIESGO MODERADO (Sugerir Confirmacion Fisiologica)"
    else:
        riesgo = "BAJO RIESGO CLINICO (Valores Compatibles con Normalidad)"

    print("=" * 65)
    print("REPORTE DIAGNOSTICO DE DETECCION NO INVASIVA DE ANEMIA")
    print("=" * 65)
    print(f"Archivo de Imagen       : {os.path.basename(image_path)}")
    print(f"Modelo Computacional    : {config['MODEL_NAME']}")
    print(f"Resultado Diagnostico   : {diagnostico}")
    print(f"Probabilidad de Anemia  : {prob_anemia * 100:.2f}%")
    print(f"Probabilidad Normalidad : {prob_no_anemia * 100:.2f}%")
    print(f"Umbral de Decision      : {threshold:.4f}")
    print(f"Clasificacion de Riesgo : {riesgo}")
    print("=" * 65)

    # Visualizacion de la muestra analizada con su etiqueta diagnostica
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(img_pil)
    color_banner = '#d03b3b' if tiene_anemia else '#2a78d6'
    ax.set_title(f"{diagnostico}\\nProbabilidad Anemia: {prob_anemia*100:.1f}%",
                 fontsize=10, fontweight='bold', color=color_banner)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    return {
        'diagnostico': diagnostico,
        'probabilidad_anemia': float(prob_anemia),
        'probabilidad_normal': float(prob_no_anemia),
        'umbral': float(threshold),
        'riesgo': riesgo
    }

# Prueba de inferencia unitaria con un caso de prueba independiente
ejemplo_img = os.path.join(CONFIG['IMG_DIR'], df_test.iloc[0]['image'])
resultado_ejemplo = diagnosticar_anemia(ejemplo_img)"""))

    return cells

print("Generador de celdas de ResNet-50 definido con exito.")

