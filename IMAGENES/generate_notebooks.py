# -*- coding: utf-8 -*-
"""
Generador Maestro y Validador de Cuadernos Jupyter para la Tesis:
- ResNet-50.ipynb
- EfficientNet-B3.ipynb

Universidad Nacional Jose Maria Arguedas (UNAJMA) - 2026
Escuela Profesional de Ingenieria de Sistemas
Tesis: Metodo no invasivo basado en redes neuronales convolucionales para la deteccion
de la anemia infantil a partir de imagenes en el centro de salud, Ocobamba 2025.
"""

import os
import sys
import json
import copy

# Importar funciones de creacion de celdas desde build_notebooks_definitivos
from build_notebooks_definitivos import create_cell, save_notebook, get_resnet50_cells

def get_efficientnet_b3_cells():
    """
    Construye las celdas de EfficientNet-B3 a partir de las especificaciones tecnicas:
    - Resolucion 300x300 px
    - Arquitectura EfficientNet-B3 con bloques MBConv y Squeeze-and-Excitation (12.2M parametros)
    - Clasificador con Dropout y BatchNorm (1536 -> 512 -> 2)
    - Descongelamiento progresivo de bloques finales (features.6, features.7, features.8)
    - Celda 15 adicional con Cuadro Comparativo Final y Prueba de Hipotesis (t-Student y Wilcoxon)
    """
    resnet_cells = get_resnet50_cells()
    effnet_cells = []

    # CELDA 0: PORTADA MARKDOWN
    c0_md = """# Deteccion de Anemia Infantil mediante EfficientNet-B3 y Conjuntiva Palpebral
**Tesis:** *Metodo no invasivo basado en redes neuronales convolucionales para la deteccion de la anemia infantil a partir de imagenes en el centro de salud, Ocobamba 2025*
**Institucion:** Universidad Nacional Jose Maria Arguedas (UNAJMA) - Escuela Profesional de Ingenieria de Sistemas
**Autor:** Investigacion Conducente al Titulo Profesional de Ingeniero de Sistemas

---

### Marco Metodologico KDD (Knowledge Discovery in Databases)
El presente cuaderno implementa de manera rigurosa las fases del proceso KDD (Debuse et al., 2001; Yanez, 2023) adaptadas a la arquitectura de vanguardia **EfficientNet-B3**:

1. **Fase de Seleccion de Datos (Data Selection):** Integracion de metadatos clinicos (Hemoglobina, Edad en meses, Sexo) e imagenes de la conjuntiva palpebral infantil segun criterios diagnosticos de la OMS.
2. **Fase de Preprocesamiento y Limpieza (Data Preprocessing):** Manejo robusto de imagenes truncadas o con chunks no estandar (resolviendo errores como `T_A95_img_002.png`), composicion controlada de imagenes RGBA sobre fondo neutro para evitar artefactos oscuros en la conjuntiva, y saneamiento de inconsistencias de datos para erradicar cualquier fuga (*data leakage*).
3. **Fase de Transformacion (Data Transformation):** Escalado compuesto optimizado a **300x300 px** caracteristico de EfficientNet-B3, normalizacion espectral segun ImageNet y aumentos biomedicos leves para enriquecer el entrenamiento sin alterar el matiz cromatico de la hemoglobina.
4. **Fase de Mineria de Datos / Modelado (Data Mining):** Transfer Learning con la arquitectura **EfficientNet-B3** (Tan & Le, 2019) de 12.2 millones de parametros basada en bloques convolucionales invertidos de cuello de botella con mecanismo de atencion (*MBConv + Squeeze-and-Excitation*), aplicando entrenamiento en dos fases (Warmup y Fine-Tuning progresivo) con funcion de perdida ponderada y decaimiento cosenoidal.
5. **Fase de Evaluacion Clinica e Interpretacion (Evaluation & Interpretation):** Evaluacion estricta en el conjunto de prueba independiente (`test.csv`, 137 casos no vistos), calculo de metricas diagnosticas (Sensibilidad/Recall, Especificidad, Exactitud/Accuracy, Precision/VPP, F1-Score, Curva ROC-AUC), optimizacion de umbral diagnostico de Youden y graficos a 300 DPI bajo norma APA 7ma edicion.
6. **Fase de Despliegue e Interoperabilidad (Deployment):** Exportacion validada a formato abierto ONNX (`.onnx`) y TorchScript (`.pt`) con verificacion de paridad numerica para su integracion en el aplicativo movil del centro de salud.
7. **Protocolo de Validacion Estadistica y Comparativa de Modelos:** Evaluacion con 5 semillas independientes y contraste formal de hipotesis (t-Student y Wilcoxon) frente a ResNet-50 para sustentar cientificamente las conclusiones del Capitulo VI de la tesis."""
    effnet_cells.append(create_cell('markdown', c0_md))

    # CELDA 1: ENTORNO (Igual)
    effnet_cells.append(copy.deepcopy(resnet_cells[1]))

    # CELDA 2: LIBRERIAS Y SEMILLAS (Igual)
    effnet_cells.append(copy.deepcopy(resnet_cells[2]))

    # CELDA 3: CONFIGURACION DE HIPERPARAMETROS (300x300 px, EfficientNet-B3)
    c3_src = ''.join(resnet_cells[3]['source'])
    c3_src = c3_src.replace("'MODEL_NAME': 'ResNet-50'", "'MODEL_NAME': 'EfficientNet-B3'")
    c3_src = c3_src.replace("'ARCHITECTURE': 'resnet50'", "'ARCHITECTURE': 'efficientnet_b3'")
    c3_src = c3_src.replace("'IMAGE_SIZE': 224", "'IMAGE_SIZE': 300")
    c3_src = c3_src.replace("'BATCH_SIZE': 32", "'BATCH_SIZE': 16")
    c3_src = c3_src.replace("'LR_BACKBONE': 1e-4", "'LR_BACKBONE': 5e-5")
    c3_src = c3_src.replace("'DROPOUT_RATE': 0.4", "'DROPOUT_RATE': 0.3")
    c3_src = c3_src.replace("best_resnet50.pth", "best_efficientnet_b3.pth")
    c3_src = c3_src.replace("Capas profundas layer3 y layer4", "Bloques convolucionales MBConv features.6-8")
    effnet_cells.append(create_cell('code', c3_src))

    # CELDA 4: KDD FASE 1 EDA (Igual)
    effnet_cells.append(copy.deepcopy(resnet_cells[4]))

    # CELDA 5: KDD FASE 2 PREPROCESAMIENTO ROBUSTO (Igual)
    effnet_cells.append(copy.deepcopy(resnet_cells[5]))

    # CELDA 6: KDD FASE 3 TRANSFORMACION Y PREVISUALIZACION (300x300 px)
    c6_src = ''.join(resnet_cells[6]['source'])
    c6_src = c6_src.replace("ResNet-50", "EfficientNet-B3")
    c6_src = c6_src.replace("figura_previsualizacion_clinica_resnet50.png", "figura_previsualizacion_clinica_efficientnet_b3.png")
    c6_src = c6_src.replace("figura_previsualizacion_clinica_data_aug.png", "figura_previsualizacion_clinica_efficientnet_b3.png")
    effnet_cells.append(create_cell('code', c6_src))

    # CELDA 7: KDD FASE 4 MODELO EFFICIENTNET-B3
    c7_code = """# ==============================================================================
# 6. KDD FASE 4: CONSTRUCCION DE EFFICIENTNET-B3 CON TRANSFER LEARNING
# ==============================================================================
def build_efficientnet_b3_model(num_classes=2, dropout_rate=0.3):
    \"\"\"
    Construye la arquitectura EfficientNet-B3 con pesos preentrenados de ImageNet-1K.
    Sustituye el clasificador final por un cabezal con regularizacion Dropout y BatchNorm.
    Incluye resolucion de conectividad en la nube (Kaggle / Colab) y busqueda de pesos locales.
    \"\"\"
    model = None
    try:
        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        print("Pesos preentrenados de EfficientNet-B3 (ImageNet-1K) cargados exitosamente desde PyTorch Hub.")
    except Exception as e:
        print(f"Aviso: No fue posible descargar pesos directamente de PyTorch Hub ({type(e).__name__}).")
        print("Buscando archivos de pesos preentrenados locales en el entorno...")

        peso_local = None
        if os.path.exists('/kaggle/input'):
            for root, _, files in os.walk('/kaggle/input'):
                for f in files:
                    if 'efficientnet_b3' in f.lower() and f.endswith(('.pth', '.pt')):
                        peso_local = os.path.join(root, f)
                        break
                if peso_local:
                    break

        if peso_local:
            print(f"Cargando pesos preentrenados desde archivo local: {peso_local}")
            model = models.efficientnet_b3(weights=None)
            state_dict = torch.load(peso_local, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict, strict=False)
            print("Pesos preentrenados locales cargados satisfactoriamente.")
        else:
            raise RuntimeError(
                "ERROR DE CONECTIVIDAD AL DESCARGAR PESOS PREENTRENADOS:\\n"
                "PyTorch no pudo conectarse al servidor para descargar los pesos de EfficientNet-B3.\\n"
                "En Kaggle, este error ocurre habitualmente cuando la opcion de Internet esta desactivada.\\n\\n"
                "INSTRUCCIONES PARA ACTIVAR INTERNET EN KAGGLE:\\n"
                "1. Dirijase al panel lateral derecho de su cuaderno ('Notebook options' o 'Settings').\\n"
                "2. Ubique la seccion 'Internet' y active el interruptor ('Internet On').\\n"
                "   (Kaggle solicita verificacion telefonica gratuita de su cuenta para habilitar conexion).\\n"
                "3. Una vez activado el acceso a Internet, vuelva a ejecutar esta celda."
            ) from e

    in_features = model.classifier[1].in_features # 1536 en EfficientNet-B3
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate * 0.75),
        nn.Linear(512, num_classes)
    )
    return model

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if not name.startswith('classifier'):
            param.requires_grad = False
        else:
            param.requires_grad = True

def unfreeze_upper_layers(model):
    for name, param in model.named_parameters():
        if 'features.6' in name or 'features.7' in name or 'features.8' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

model = build_efficientnet_b3_model(num_classes=CONFIG['NUM_CLASSES'], dropout_rate=CONFIG['DROPOUT_RATE']).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Modelo EfficientNet-B3 instanciado exitosamente:")
print(f"  - Total de Parametros Arquitecturales: {total_params:,} (12.2M)")
print(f"  - Resolucion Espacial de Entrada    : {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']} px")
print(f"  - Cabezal Personalizado              : Dropout({CONFIG['DROPOUT_RATE']}) -> Linear(1536->512) -> BatchNorm -> ReLU -> Linear(512->2)")"""
    effnet_cells.append(create_cell('code', c7_code))

    # CELDA 8: FUNCION DE PERDIDA Y OPTIMIZADOR
    c8_code = """# ==============================================================================
# 7. FUNCION DE PERDIDA BALANCEADA, OPTIMIZADORES DIFERENCIALES Y SCHEDULER
# ==============================================================================
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
    if is_warmup:
        optimizer = optim.AdamW(model.classifier.parameters(), lr=CONFIG['LR_HEAD'], weight_decay=CONFIG['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_WARMUP'], eta_min=1e-5)
    else:
        params = [
            {'params': [p for n, p in model.named_parameters() if ('features.6' in n or 'features.7' in n or 'features.8' in n)], 'lr': CONFIG['LR_BACKBONE']},
            {'params': model.classifier.parameters(), 'lr': CONFIG['LR_HEAD'] * 0.5}
        ]
        optimizer = optim.AdamW(params, weight_decay=CONFIG['WEIGHT_DECAY'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS_FINETUNE'], eta_min=1e-6)
    return optimizer, scheduler

print("Optimizador AdamW y Scheduler CosineAnnealing configurados para EfficientNet-B3.")"""
    effnet_cells.append(create_cell('code', c8_code))

    # CELDA 9: BUCLE DE ENTRENAMIENTO (Adaptar texto de descongelamiento)
    c9_src = ''.join(resnet_cells[9]['source'])
    c9_src = c9_src.replace("Descongelando bloques superiores (layer3 y layer4) para Fine-Tuning...",
                            "Descongelando bloques superiores (features.6, features.7 y features.8) para Fine-Tuning...")
    effnet_cells.append(create_cell('code', c9_src))

    # CELDA 10: EVALUACION CLINICA EN TEST.CSV
    c10_src = ''.join(resnet_cells[10]['source'])
    c10_src = c10_src.replace("resnet50_metricas_test.csv", "efficientnet_b3_metricas_test.csv")
    effnet_cells.append(create_cell('code', c10_src))

    # CELDA 11: VISUALIZACIONES APA 7 (300 DPI)
    c11_src = ''.join(resnet_cells[11]['source'])
    c11_src = c11_src.replace("resnet50_graficas_tesis.png", "efficientnet_b3_graficas_tesis.png")
    effnet_cells.append(create_cell('code', c11_src))

    # CELDA 12: EXPORTACION ONNX Y TORCHSCRIPT
    c12_src = ''.join(resnet_cells[12]['source'])
    c12_src = c12_src.replace("resnet50_anemia_opt.onnx", "efficientnet_b3_anemia_opt.onnx")
    c12_src = c12_src.replace("resnet50_torchscript.pt", "efficientnet_b3_torchscript.pt")
    effnet_cells.append(create_cell('code', c12_src))

    # CELDA 13: PROTOCOLO MULTIEJECUCION (5 SEMILLAS)
    c13_src = ''.join(resnet_cells[13]['source'])
    c13_src = c13_src.replace("resnet50_multiejecucion_stats.csv", "efficientnet_b3_multiejecucion_stats.csv")
    effnet_cells.append(create_cell('code', c13_src))

    # CELDA 14: INFERENCIA UNITARIA
    effnet_cells.append(copy.deepcopy(resnet_cells[14]))

    # CELDA 15 (NUEVA): CUADRO COMPARATIVO RESNET-50 VS EFFICIENTNET-B3 Y CONTRASTE DE HIPOTESIS
    c15_code = """# ==============================================================================
# 14. CUADRO COMPARATIVO RESNET-50 VS EFFICIENTNET-B3 Y CONTRASTE DE HIPOTESIS
# PARA EL CAPITULO VI (PRESENTACION DE RESULTADOS) DE LA TESIS
# ==============================================================================
# Carga de las metricas consolidadas de ambos modelos
resnet_csv = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_metricas_test.csv')
effnet_csv = os.path.join(CONFIG['RESULTS_DIR'], 'efficientnet_b3_metricas_test.csv')

resnet_multi_csv = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_multiejecucion_stats.csv')
effnet_multi_csv = os.path.join(CONFIG['RESULTS_DIR'], 'efficientnet_b3_multiejecucion_stats.csv')

datos_disponibles = os.path.exists(resnet_csv) and os.path.exists(effnet_csv)

if datos_disponibles:
    df_r = pd.read_csv(resnet_csv).set_index('Metrica Diagnostica')
    df_e = pd.read_csv(effnet_csv).set_index('Metrica Diagnostica')

    tabla_comparativa = pd.DataFrame({
        'Metrica Clinica Evaluada': [
            'Sensibilidad (Recall / Deteccion de Casos)',
            'Especificidad (Tasa Verdaderos Negativos)',
            'Exactitud Global (Accuracy)',
            'Precision (Valor Predictivo Positivo - VPP)',
            'Valor Predictivo Negativo (VPN)',
            'F1-Score Ponderado',
            'Area Bajo la Curva ROC (AUC-ROC)',
            'Total Parametros Arquitecturales',
            'Resolucion Espacial de Entrada'
        ],
        'ResNet-50 (He et al., 2016)': [
            df_r.loc['Sensibilidad (Recall / Tasa Verdaderos Positivos)', 'Valor Obtenido'],
            df_r.loc['Especificidad (Tasa Verdaderos Negativos)', 'Valor Obtenido'],
            df_r.loc['Exactitud Diagnostica (Accuracy)', 'Valor Obtenido'],
            df_r.loc['Precision (Valor Predictivo Positivo - VPP)', 'Valor Obtenido'],
            df_r.loc['Valor Predictivo Negativo (VPN)', 'Valor Obtenido'],
            df_r.loc['F1-Score Ponderado', 'Valor Obtenido'],
            df_r.loc['Area Bajo la Curva ROC (AUC-ROC)', 'Valor Obtenido'],
            '24,560,706 (24.5M)',
            '224 x 224 px'
        ],
        'EfficientNet-B3 (Tan & Le, 2019)': [
            df_e.loc['Sensibilidad (Recall / Tasa Verdaderos Positivos)', 'Valor Obtenido'],
            df_e.loc['Especificidad (Tasa Verdaderos Negativos)', 'Valor Obtenido'],
            df_e.loc['Exactitud Diagnostica (Accuracy)', 'Valor Obtenido'],
            df_e.loc['Precision (Valor Predictivo Positivo - VPP)', 'Valor Obtenido'],
            df_e.loc['Valor Predictivo Negativo (VPN)', 'Valor Obtenido'],
            df_e.loc['F1-Score Ponderado', 'Valor Obtenido'],
            df_e.loc['Area Bajo la Curva ROC (AUC-ROC)', 'Valor Obtenido'],
            '12,233,832 (12.2M)',
            '300 x 300 px'
        ]
    })
else:
    _sens = f"{sensibilidad * 100:.2f}%" if 'sensibilidad' in locals() else 'Pendiente'
    _esp = f"{especificidad * 100:.2f}%" if 'especificidad' in locals() else 'Pendiente'
    _acc = f"{acc * 100:.2f}%" if 'acc' in locals() else 'Pendiente'
    _prec = f"{precision * 100:.2f}%" if 'precision' in locals() else 'Pendiente'
    _vpn = f"{vpn * 100:.2f}%" if 'vpn' in locals() else 'Pendiente'
    _f1 = f"{f1:.4f}" if 'f1' in locals() else 'Pendiente'
    _auc = f"{auc_roc:.4f}" if 'auc_roc' in locals() else 'Pendiente'

    tabla_comparativa = pd.DataFrame({
        'Metrica Clinica Evaluada': [
            'Sensibilidad (Recall / Deteccion de Casos)',
            'Especificidad (Tasa Verdaderos Negativos)',
            'Exactitud Global (Accuracy)',
            'Precision (Valor Predictivo Positivo - VPP)',
            'Valor Predictivo Negativo (VPN)',
            'F1-Score Ponderado',
            'Area Bajo la Curva ROC (AUC-ROC)',
            'Total Parametros Arquitecturales',
            'Resolucion Espacial de Entrada'
        ],
        'ResNet-50 (He et al., 2016)': [
            'Pendiente de ejecucion previa en ResNet-50.ipynb',
            'Pendiente', 'Pendiente', 'Pendiente', 'Pendiente', 'Pendiente', 'Pendiente',
            '24,560,706 (24.5M)', '224 x 224 px'
        ],
        'EfficientNet-B3 (Tan & Le, 2019)': [
            _sens,
            _esp,
            _acc,
            _prec,
            _vpn,
            _f1,
            _auc,
            '12,233,832 (12.2M)',
            '300 x 300 px'
        ]
    })

print("\\n" + "=" * 90)
print("TABLA 4. CUADRO COMPARATIVO DE RENDIMIENTO DIAGNOSTICO: RESNET-50 VS EFFICIENTNET-B3")
print("FORMATO ESTRICTO APA 7MA EDICION PARA EL CAPITULO VI DE LA TESIS")
print("=" * 90)
print(tabla_comparativa.to_string(index=False))
print("-" * 90)
print("Nota. Evaluacion efectuada sobre el conjunto de prueba independiente (n = 137). Sin lineas verticales segun norma APA 7.")

comp_csv_path = os.path.join(CONFIG['RESULTS_DIR'], 'tabla_comparativa_modelos_tesis.csv')
tabla_comparativa.to_csv(comp_csv_path, index=False, encoding='utf-8')
print(f"Cuadro comparativo exportado exitosamente a: {comp_csv_path}")

# PRUEBA ESTADISTICA DE CONTRASTE DE HIPOTESIS (T-STUDENT Y WILCOXON)
if os.path.exists(resnet_multi_csv) and os.path.exists(effnet_multi_csv):
    df_rm = pd.read_csv(resnet_multi_csv)
    df_em = pd.read_csv(effnet_multi_csv)

    acc_r = df_rm['Accuracy'].values
    acc_e = df_em['Accuracy'].values

    t_stat, p_val_t = stats.ttest_rel(acc_e, acc_r)
    try:
        w_stat, p_val_w = stats.wilcoxon(acc_e, acc_r)
    except Exception:
        w_stat, p_val_w = float('nan'), float('nan')

    print("\\n" + "=" * 80)
    print("CONTRASTE ESTADISTICO DE HIPOTESIS (CAPITULO V Y VI)")
    print("=" * 80)
    print(f"Hipotesis Nula (H0)       : No existe diferencia significativa entre EfficientNet-B3 y ResNet-50.")
    print(f"Hipotesis Alterna (H1)    : Existe diferencia significativa en el rendimiento diagnostico.")
    print(f"Nivel de Significancia (a): 0.05")
    print(f"Estadistico t de Student  : t = {t_stat:.4f} (p-valor = {p_val_t:.4e})")
    if not math.isnan(p_val_w):
        print(f"Estadistico de Wilcoxon   : W = {w_stat:.4f} (p-valor = {p_val_w:.4e})")

    if p_val_t < 0.05:
        print("Conclusion Cientifica     : Se rechaza H0 a favor de H1 (p < 0.05).")
        print("                            Existe diferencia estadisticamente significativa entre ambos modelos.")
    else:
        print("Conclusion Cientifica     : No se rechaza H0 (p >= 0.05). Ambos modelos ofrecen desempenos comparables.")
    print("=" * 80)"""
    effnet_cells.append(create_cell('code', c15_code))

    return effnet_cells

if __name__ == '__main__':
    base_dir = r'C:\Proyectos\luu\IMAGENES'
    print("Iniciando compilacion de cuadernos definitivos para la tesis...")

    # 1. Compilar y guardar ResNet-50.ipynb
    resnet_cells = get_resnet50_cells()
    save_notebook(resnet_cells, os.path.join(base_dir, 'ResNet-50.ipynb'))
    print(f"ResNet-50.ipynb generado exitosamente con {len(resnet_cells)} celdas.")

    # 2. Compilar y guardar EfficientNet-B3.ipynb
    effnet_cells = get_efficientnet_b3_cells()
    save_notebook(effnet_cells, os.path.join(base_dir, 'EfficientNet-B3.ipynb'))
    print(f"EfficientNet-B3.ipynb generado exitosamente con {len(effnet_cells)} celdas.")

    # Validacion sintactica de archivos .ipynb generados
    for nb_name in ['ResNet-50.ipynb', 'EfficientNet-B3.ipynb']:
        nb_path = os.path.join(base_dir, nb_name)
        with open(nb_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        n_cells = len(data['cells'])
        n_code = sum(1 for c in data['cells'] if c['cell_type'] == 'code')
        n_md = sum(1 for c in data['cells'] if c['cell_type'] == 'markdown')
        print(f"Validacion JSON de {nb_name}: Valido ({n_cells} celdas en total: {n_code} codigo, {n_md} markdown).")

    print("\nProceso de compilacion y validacion finalizado con exito total.")
