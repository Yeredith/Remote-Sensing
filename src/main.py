import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import csv
import numpy as np
from data_utils import TrainsetFromFolder, ValsetFromFolder, TestsetFromFolder, TestsetFromFolder_Noisy
from models import SFCSR, MCNet, Propuesto , Propuesto2, Modificacion1, Modificacion2
from scipy.io import savemat

# Configuración de CuDNN para un mejor rendimiento
torch.backends.cudnn.benchmark = True

# Clase para convertir diccionarios en objetos con atributos
class ConfigNamespace:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# Configuración desde JSON
def load_config():
    json_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(json_path, 'r') as file:
        return json.load(file)

# Selección de modelo
def select_model(config, model_name):
    model_config = ConfigNamespace(config["models"][model_name])
    model_config.cuda = config.get("cuda", False)  # Asegurar atributo cuda

    if model_name == "SFCSR":
        return SFCSR(model_config)
    if model_name == "MCNet":
        return MCNet(model_config)
    if model_name == "Propuesto":
        return Propuesto(model_config)
    if model_name == "Propuesto2":
        return Propuesto2(model_config)
    if model_name == "Modificacion1":
        return Modificacion1(model_config)
    if model_name == "Modificacion2":
        return Modificacion2(model_config)
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

# Configurar rutas de salida
def setup_output_paths(config, model_name):
    try:
        # Definir rutas base
        base_path = config["output"]["results_path"]
        model_path = os.path.join(base_path, model_name)
        checkpoints_path = os.path.join(model_path, "checkpoints")
        metrics_path = os.path.join(model_path, "metrics")
        params_csv_path = os.path.join(model_path, "params.csv")

        # Crear directorios si no existen
        os.makedirs(checkpoints_path, exist_ok=True)
        os.makedirs(metrics_path, exist_ok=True)

        # Ruta de métricas separada para normal y noise
        metrics_csv_path = os.path.join(metrics_path, "metrics.csv")

        return checkpoints_path, metrics_csv_path, params_csv_path
    except Exception as e:
        raise ValueError(f"Error al configurar las rutas de salida. Detalles: {e}")


def train(train_loader, model, optimizer, criterion, device, model_name):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Entrenando"):
        # batch[0] -> LR, batch[1] -> HR
        # Ambas ya están con forma (batch_size, channels, height, width)
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # Validar dimensiones (se esperan 4: [B, C, H, W])
        if len(inputs.size()) != 4 or len(labels.size()) != 4:
            raise ValueError("Las dimensiones de 'inputs' y 'labels' no son correctas. "
                             "Se esperaban tensores con 4 dimensiones (B, C, H, W).")

        # Reiniciar gradientes
        optimizer.zero_grad()

        if model_name == "MCNet":
            # MCNet procesa todas las bandas a la vez
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            total_loss += loss.item()
        else:
            # Lógica para modelos SFCSR y similares
            # (la idea original parece procesar bandas una por una,
            #  pero aquí se deja tal cual para conservar la estructura)
            batch_loss = 0

            # Recorremos cada banda en el eje "channels" (dim=1)
            for i in range(inputs.size(1)):
                if i == 0:
                    x = inputs[:, 0:3, :, :]
                    new_label = labels[:, 0:3, :, :]
                elif i == inputs.size(1) - 1:
                    x = inputs[:, i-2:i+1, :, :]
                    new_label = labels[:, i-2:i+1, :, :]
                else:
                    x = inputs[:, i-1:i+2, :, :]
                    new_label = labels[:, i-1:i+2, :, :]

                # Forward y retropropagación
                output = model(x)
                loss = criterion(output, new_label)
                loss.backward()
                batch_loss += loss.item()

            total_loss += batch_loss

        optimizer.step()

    return total_loss / len(train_loader)

def val(val_loader, model, criterion, device, model_name):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando"):
            # batch[0] = inputs (LR), batch[1] = labels (HR)
            # ya en forma (B, C, H, W)
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if model_name == "MCNet":
                # MCNet procesa todas las bandas a la vez
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            else:
                # Lógica para modelos SFCSR y similares
                batch_loss = 0
                for i in range(inputs.size(1)):  # Procesar por banda
                    if i == 0:
                        x = inputs[:, 0:3, :, :]
                        new_label = labels[:, 0:3, :, :]
                    elif i == inputs.size(1) - 1:
                        x = inputs[:, i - 2:i + 1, :, :]
                        new_label = labels[:, i - 2:i + 1, :, :]
                    else:
                        x = inputs[:, i - 1:i + 2, :, :]
                        new_label = labels[:, i - 1:i + 2, :, :]

                    output = model(x)
                    loss = criterion(output, new_label)
                    batch_loss += loss.item()

                total_loss += batch_loss

    return total_loss / len(val_loader)



def save_checkpoint(model, optimizer, checkpoints_path, epoch, data_type="normal"):

    #Guarda el checkpoint del modelo con un sufijo que indica el tipo de datos.

    try:
        # Validar el tipo de datos
        if data_type not in ["normal", "noise"]:
            raise ValueError(f"data_type inválido: {data_type}. Debe ser 'normal' o 'noise'.")

        # Construir la ruta de salida
        model_out_path = os.path.join(checkpoints_path, f"model_epoch_{epoch}_{data_type}.pth")
        os.makedirs(checkpoints_path, exist_ok=True)

        # Crear el estado del checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        # Guardar el estado
        torch.save(state, model_out_path)
        print(f"Checkpoint guardado: {model_out_path}")

    except Exception as e:
        print(f"Error al guardar el checkpoint en {checkpoints_path}. Archivo: {model_out_path}")
        print(f"Detalles del error: {e}")


def save_metrics_to_csv(csv_path, loss_values, val_loss_values, data_type="normal"):
    try:
        # Validar el tipo de datos
        if data_type not in ["normal", "noise"]:
            raise ValueError(f"data_type inválido: {data_type}. Debe ser 'normal' o 'noise'.")

        # Construir la ruta del archivo CSV
        csv_path_with_type = f"{os.path.splitext(csv_path)[0]}_{data_type}.csv"

        # Determinar el modo de apertura (append si existe, write si no existe)
        file_exists = os.path.exists(csv_path_with_type)
        mode = 'a' if file_exists else 'w'

        # Escribir las métricas en el archivo CSV
        with open(csv_path_with_type, mode=mode, newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Escribir encabezado solo si el archivo no existe
            if not file_exists:
                writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

            # Escribir las métricas por época
            for epoch, (train_loss, val_loss) in enumerate(zip(loss_values, val_loss_values), start=1):
                writer.writerow([epoch, train_loss, val_loss])

        print(f"Métricas guardadas en: {csv_path_with_type}")
    except Exception as e:
        print(f"Error al guardar las métricas en {csv_path_with_type}. Detalles del error: {e}")




import matplotlib.pyplot as plt

def load_last_checkpoint(model, optimizer, checkpoints_path, mode):

    if not os.path.exists(checkpoints_path):
        print(f"No se encontró el directorio de checkpoints: {checkpoints_path}. Iniciando desde el principio.")
        return model, optimizer, 0

    # Obtener lista de checkpoints que coincidan 
    if mode == 'normal':
        checkpoint_files = [
            f for f in os.listdir(checkpoints_path) 
            if f.startswith("model_epoch_") and f.endswith(f"_{mode}.pth")
        ]
    elif mode == 'noise':
         checkpoint_files = [
            f for f in os.listdir(checkpoints_path) 
            if f.startswith("model_epoch_") and f.endswith(f"_{mode}.pth")
        ]

    if not checkpoint_files:
        print(f"No se encontraron checkpoints en {checkpoints_path}. Iniciando desde el principio.")
        return model, optimizer, 0

    # Ordenar por el número de época extraído del nombre del archivo
    try:
        checkpoint_files = sorted(
            checkpoint_files,
            key=lambda x: int(x.split("_")[2])  # Extrae el número de época
        )
    except ValueError as e:
        print(f"Error al procesar nombres de checkpoint: {e}")
        return model, optimizer, 0

    last_checkpoint_path = os.path.join(checkpoints_path, checkpoint_files[-1])
    try:
        print(f"Cargando el último checkpoint: {last_checkpoint_path}")
        checkpoint = torch.load(last_checkpoint_path)

        # Verificar que el checkpoint contiene las claves esperadas
        if "model" not in checkpoint or "optimizer" not in checkpoint or "epoch" not in checkpoint:
            raise ValueError("El archivo de checkpoint no contiene todas las claves necesarias ('model', 'optimizer', 'epoch').")

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]

        print(f"Checkpoint {mode} cargado correctamente. Comenzando desde la época {start_epoch + 1}.")
        return model, optimizer, start_epoch
    except Exception as e:
        print(f"Error al cargar el checkpoint desde {last_checkpoint_path}: {e}")
        return model, optimizer, 0


from scipy.io import savemat  # Importar savemat para guardar imágenes en formato .mat
from eval import EPI, SSIM, PSNR

from torch.cuda.amp import autocast

def test_model(test_loader, model, model_name, device, test_path, data_type=None):
    model.eval()
    os.makedirs(test_path, exist_ok=True)

    per_image_csv_path = os.path.join(test_path, f"metrics_{data_type}.csv")
    overall_metrics_csv_path = os.path.join(test_path, f"overall_metrics_{data_type}.csv")

    # Crear encabezado para métricas por imagen
    with open(per_image_csv_path, "w", newline="") as per_image_csv:
        writer = csv.writer(per_image_csv)
        writer.writerow(["Filename", "PSNR", "SSIM", "EPI", "Time (ms)"])

    overall_metrics = {"PSNR": [], "SSIM": [], "EPI": [], "Time": []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Testeando {model_name} ({data_type})")):
            inputs, labels, filenames = batch[0].to(device), batch[1].to(device), batch[2]

            sub_batch_size = 1  # Procesar una imagen a la vez
            total_samples = inputs.size(0)

            for start_idx in range(0, total_samples, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, total_samples)

                # Extraer sub-batch
                sub_inputs = inputs[start_idx:end_idx]
                sub_labels = labels[start_idx:end_idx]
                sub_filenames = filenames[start_idx:end_idx]

                # Medimos tiempo de inferencia por sub-batch
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()

                # Forward con precisión mixta
                with autocast():
                    sub_outputs = model(sub_inputs)

                end_time.record()
                torch.cuda.synchronize()  # Sincronizar tiempo GPU
                elapsed_time = start_time.elapsed_time(end_time)

                # Convertir a numpy
                sub_outputs_np = sub_outputs.cpu().numpy().transpose(0, 2, 3, 1)
                sub_labels_np = sub_labels.cpu().numpy().transpose(0, 2, 3, 1)

                # Calcular métricas para cada imagen del sub-batch
                for i in range(sub_outputs_np.shape[0]):
                    output_image = sub_outputs_np[i]
                    label_image = sub_labels_np[i]
                    filename = sub_filenames[i]

                    psnr_per_channel = []
                    ssim_per_channel = []
                    epi_per_channel = []

                    for channel in range(output_image.shape[-1]):
                        output_2d = output_image[..., channel]
                        label_2d = label_image[..., channel]

                        psnr_channel = PSNR(output_2d, label_2d)
                        psnr_per_channel.append(psnr_channel)

                        try:
                            ssim_channel = SSIM(output_2d, label_2d)
                            ssim_per_channel.append(ssim_channel)
                        except Exception as e:
                            print(f"Error calculando SSIM para el canal {channel} en {filename}: {e}")
                            ssim_per_channel.append(0)

                        try:
                            epi_channel = EPI(output_2d[np.newaxis, ...], label_2d[np.newaxis, ...])
                            epi_per_channel.append(epi_channel)
                        except Exception as e:
                            print(f"Error calculando EPI para el canal {channel} en {filename}: {e}")
                            epi_per_channel.append(0)

                    # Promedio de métricas por canal
                    psnr = float(np.mean(psnr_per_channel))
                    ssim = float(np.mean(ssim_per_channel))
                    epi = float(np.mean(epi_per_channel))

                    # Escribir métricas por imagen
                    with open(per_image_csv_path, "a", newline="") as per_image_csv:
                        writer = csv.writer(per_image_csv)
                        writer.writerow([filename, psnr, ssim, epi, elapsed_time])

                    # Guardar métricas globales
                    overall_metrics["PSNR"].append(psnr)
                    overall_metrics["SSIM"].append(ssim)
                    overall_metrics["EPI"].append(epi)
                    overall_metrics["Time"].append(elapsed_time)

                    # Guardar imagen generada en .mat
                    output_file = os.path.join(test_path, f"{os.path.splitext(filename)[0]}_output.mat")
                    savemat(output_file, {'generated': output_image, 'ground_truth': label_image})
                    print(f"Imagen guardada en {output_file}")

                # Limpiar memoria después de cada sub-batch
                torch.cuda.empty_cache()

    # Guardar métricas generales
    with open(overall_metrics_csv_path, "w", newline="") as overall_csv:
        writer = csv.writer(overall_csv)
        writer.writerow(["Metric", "Average Value"])
        writer.writerow(["Average PSNR", float(np.mean(overall_metrics["PSNR"]))])
        writer.writerow(["Average SSIM", float(np.mean(overall_metrics["SSIM"]))])
        writer.writerow(["Average EPI", float(np.mean(overall_metrics["EPI"]))])
        writer.writerow(["Average Time (ms)", float(np.mean(overall_metrics["Time"]))])

    print(f"Pruebas completadas para {model_name} ({data_type}).")
    print(f"Resultados guardados en {per_image_csv_path} y {overall_metrics_csv_path}.")

    
def save_model_params_to_csv(params_csv_path, model_name, model, data_type="normal"):
    try:
        # Calcular el total de parámetros y los parámetros entrenables
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Guardar los datos en el archivo CSV
        file_exists = os.path.exists(params_csv_path)
        with open(params_csv_path, mode='a' if file_exists else 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Escribir el encabezado solo si el archivo no existe
            if not file_exists:
                writer.writerow(["Model Name", "Data Type", "Total Parameters", "Trainable Parameters"])
            writer.writerow([model_name, data_type, total_params, trainable_params])
        print(f"Parámetros del modelo guardados en: {params_csv_path}")
    except Exception as e:
        print(f"Error al guardar los parámetros del modelo en {params_csv_path}: {e}")

    
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import csv
import numpy as np


# Otras utilidades
from scipy.io import savemat
from eval import EPI, SSIM, PSNR
import copy

def main():
    try:

        config = load_config()


        available_gpus = list(range(torch.cuda.device_count()))
        if not available_gpus and config["cuda"]:
            raise EnvironmentError("CUDA habilitada, pero no se detectaron GPUs disponibles.")

        invalid_gpus = [gpu_id for gpu_id in config["gpu"]["gpu_ids"] if gpu_id not in available_gpus]
        if invalid_gpus:
            raise ValueError(f"IDs de GPU inválidos: {invalid_gpus}. GPUs disponibles: {available_gpus}")

        device = torch.device("cuda" if config["cuda"] else "cpu")
        print(f"Entrenando en: {device}")


        for model_name in config["model_list"]:
            print(f"\n=== Iniciando entrenamiento para el modelo: {model_name} ===")


            train_normal_dataset = TrainsetFromFolder(config["training"]["train_data"]["normal"], config["database"]["image_bands"])
            train_normal_loader = DataLoader(
                train_normal_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=True,
                pin_memory=True
            )

            train_noise_dataset = TrainsetFromFolder(config["training"]["train_data"]["noise"], config["database"]["image_bands"])
            train_noise_loader = DataLoader(
                train_noise_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=True,
                pin_memory=True
            )

            val_dataset = ValsetFromFolder(config["training"]["val_data"]["normal"], config["database"]["image_bands"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                pin_memory=True
            )

            test_dataset_normal = TestsetFromFolder(config["test"]["test_data"]["normal"], config["database"]["image_bands"])
            test_loader_normal = DataLoader(
                test_dataset_normal,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                pin_memory=True
            )

            test_dataset_noise = TestsetFromFolder_Noisy(config["test"]["test_data"]["noise"], config["database"]["image_bands"])
            test_loader_noise = DataLoader(
                test_dataset_noise,
                batch_size=config["training"]["batch_size"],
                shuffle=False,
                pin_memory=True
            )


            model = select_model(config, model_name)
            if config["gpu"]["use_multi_gpu"] and len(config["gpu"]["gpu_ids"]) > 1:
                model = nn.DataParallel(model, device_ids=config["gpu"]["gpu_ids"])
            model = model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
            criterion = nn.L1Loss()


            initial_model_state = copy.deepcopy(model.state_dict())
            initial_optimizer_state = copy.deepcopy(optimizer.state_dict())


            checkpoints_path, csv_path, params_csv_path = setup_output_paths(config, model_name)
            test_path_normal = os.path.join(config["output"]["results_path"], model_name, "test_results_normal")
            test_path_noise = os.path.join(config["output"]["results_path"], model_name, "test_results_noise")
            os.makedirs(test_path_normal, exist_ok=True)
            os.makedirs(test_path_noise, exist_ok=True)


            save_model_params_to_csv(params_csv_path, model_name, model, data_type="normal")


            model, optimizer, start_epoch_normal = load_last_checkpoint(model, optimizer, checkpoints_path, mode="normal")
            model, optimizer, start_epoch_noise = load_last_checkpoint(model, optimizer, checkpoints_path, mode="noise")


            model.load_state_dict(initial_model_state)
            optimizer.load_state_dict(initial_optimizer_state)
            start_epoch_normal = 0

            if start_epoch_normal < config["training"]["epochs"]:
                train_loss_values_normal = []
                val_loss_values_normal = []

                print(f"\n--- Entrenamiento con imágenes normales (modelo {model_name}) ---")
                for epoch in range(start_epoch_normal + 1, config["training"]["epochs"] + 1):
                    print(f"Epoch {epoch}/{config['training']['epochs']} (normal)")
                    train_loss = train(train_normal_loader, model, optimizer, criterion, device, model_name)
                    val_loss = val(val_loader, model, criterion, device, model_name)

                    print(f"Train Loss (normal): {train_loss:.6f}, Val Loss (normal): {val_loss:.6f}")
                    train_loss_values_normal.append(train_loss)
                    val_loss_values_normal.append(val_loss)

                    # Guardar checkpoint y métricas
                    save_checkpoint(model, optimizer, checkpoints_path, epoch, "normal")
                    save_metrics_to_csv(csv_path, [train_loss], [val_loss], "normal")


            model.load_state_dict(initial_model_state)
            optimizer.load_state_dict(initial_optimizer_state)
            start_epoch_noise = 0


            if start_epoch_noise < config["training"]["epochs"]:
                train_loss_values_noise = []
                val_loss_values_noise = []

                print(f"\n--- Entrenamiento con imágenes con ruido (modelo {model_name}) ---")
                for epoch in range(start_epoch_noise + 1, config["training"]["epochs"] + 1):
                    print(f"Epoch {epoch}/{config['training']['epochs']} (noise)")
                    train_loss = train(train_noise_loader, model, optimizer, criterion, device, model_name)
                    val_loss = val(val_loader, model, criterion, device, model_name)

                    print(f"Train Loss (noise): {train_loss:.6f}, Val Loss (noise): {val_loss:.6f}")
                    train_loss_values_noise.append(train_loss)
                    val_loss_values_noise.append(val_loss)

                    save_checkpoint(model, optimizer, checkpoints_path, epoch, "noise")
                    save_metrics_to_csv(csv_path, [train_loss], [val_loss], "noise")


            print(f"\n=== Evaluación en test normal (modelo {model_name}) ===")
            test_model(test_loader_normal, model, model_name, device, test_path_normal, data_type="normal")

            print(f"\n=== Evaluación en test noise (modelo {model_name}) ===")
            test_model(test_loader_noise, model, model_name, device, test_path_noise, data_type="noise")

            print(f"Entrenamiento y evaluación completados para el modelo: {model_name}")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()