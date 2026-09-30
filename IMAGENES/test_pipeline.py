import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms

def test_notebooks_json():
    print("--- 1. Validando estructura JSON de los notebooks ---")
    for nb_name in ['ResNet-50.ipynb', 'EfficientNet-B3.ipynb']:
        path = os.path.join(r'C:\Proyectos\luu\IMAGENES', nb_name)
        assert os.path.exists(path), f"El archivo {nb_name} no existe"
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'cells' in data, f"No hay celdas en {nb_name}"
        code_cells = [c for c in data['cells'] if c['cell_type'] == 'code']
        md_cells = [c for c in data['cells'] if c['cell_type'] == 'markdown']
        print(f"  [OK] {nb_name}: {len(data['cells'])} celdas totales ({len(code_cells)} código, {len(md_cells)} markdown).")

def test_data_pipeline():
    print("\n--- 2. Validando carga de datos e imágenes ---")
    dataset_dir = r'C:\Proyectos\luu\IMAGENES\Dataset'
    images_dir = os.path.join(dataset_dir, 'images')
    train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))

    first_img_name = train_df.iloc[0]['image']
    first_img_path = os.path.join(images_dir, first_img_name)
    assert os.path.exists(first_img_path), f"Imagen no encontrada: {first_img_path}"

    img = Image.open(first_img_path).convert('RGB')
    print(f"  [OK] Imagen de prueba '{first_img_name}' cargada exitosamente. Dimensiones originales: {img.size}")

    # Probar transformaciones
    t224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    t300 = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tensor224 = t224(img).unsqueeze(0)
    tensor300 = t300(img).unsqueeze(0)
    assert tensor224.shape == (1, 3, 224, 224), f"Forma errónea para ResNet: {tensor224.shape}"
    assert tensor300.shape == (1, 3, 300, 300), f"Forma errónea para EfficientNet: {tensor300.shape}"
    print(f"  [OK] Transformación ResNet (1, 3, 224, 224) correcta.")
    print(f"  [OK] Transformación EfficientNet (1, 3, 300, 300) correcta.")
    return tensor224, tensor300

def test_models_forward(tensor224, tensor300):
    print("\n--- 3. Validando paso hacia adelante (Forward Pass) de modelos ---")

    # ResNet-50
    model_resnet = models.resnet50(weights=None)
    model_resnet.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 2)
    )
    model_resnet.eval()
    with torch.no_grad():
        out_resnet = model_resnet(tensor224)
    assert out_resnet.shape == (1, 2), f"Salida errónea en ResNet: {out_resnet.shape}"
    probs_resnet = torch.softmax(out_resnet, dim=1)
    print(f"  [OK] ResNet-50 salida shape: {out_resnet.shape} | Probabilidades: {probs_resnet.numpy()[0]}")

    # EfficientNet-B3
    model_eff = models.efficientnet_b3(weights=None)
    model_eff.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1536, 256),
        torch.nn.SiLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 2)
    )
    model_eff.eval()
    with torch.no_grad():
        out_eff = model_eff(tensor300)
    assert out_eff.shape == (1, 2), f"Salida errónea en EfficientNet: {out_eff.shape}"
    probs_eff = torch.softmax(out_eff, dim=1)
    print(f"  [OK] EfficientNet-B3 salida shape: {out_eff.shape} | Probabilidades: {probs_eff.numpy()[0]}")

if __name__ == '__main__':
    test_notebooks_json()
    t224, t300 = test_data_pipeline()
    test_models_forward(t224, t300)
    print("\n========================================================")
    print("  TODAS LAS PRUEBAS UNITARIAS PASARON SATISFACTORIAMENTE")
    print("========================================================")
