clc
clear
close all

%% Definir hiperparámetros
upscale_factor = 4;
data_type = 'Original'; % Tipo de datos

%% Rutas de salida
baseOutputFolder = 'F:\SR-RS\Remote Sensing\Preprocessing\Data';
testFolder = fullfile(baseOutputFolder, 'Original', data_type, num2str(upscale_factor));

if ~exist(testFolder, 'dir')
    mkdir(testFolder);
end


%% Rutas de datos fuente
testPath = 'F:\Remote Sensing\Test_HR\val_1st\HR'; % Imágenes de prueba

%% Procesar imágenes de prueba
disp('Procesando imágenes de prueba...');
testFiles = dir(fullfile(testPath, '*.jpg')); % Asumiendo formato JPG

for index = 1:length(testFiles)
    % Cargar imagen
    imgName = testFiles(index).name;
    imgPath = fullfile(testPath, imgName);
    img = imread(imgPath);

    % Normalizar imagen
    imgDouble = double(img);
    imgNorm = imgDouble / max(imgDouble(:)); % Normalizar entre 0 y 1

    % Recortar y redimensionar para imágenes HR y LR
    HR = modcrop(imgNorm, upscale_factor); % Recorte para divisibilidad
    LR = imresize(HR, 1 / upscale_factor); % Imagen de baja resolución

    % Guardar en la carpeta de Test
    [~, name, ~] = fileparts(imgName);
    save(fullfile(testFolder, [name, '.mat']), 'HR', 'LR');
end

disp('Procesamiento completado.');

clc
clear
close all

%% Definir hiperparámetros
upscale_factor = 4;
data_type = 'Remote/Noise'; % Tipo de datos
noise_dB = 17; % Nivel de ruido en dB

%% Rutas de salida
baseOutputFolder = 'F:\SR-RS\Remote Sensing\Preprocessing\Data';
testFolder = fullfile(baseOutputFolder, 'test', data_type, num2str(upscale_factor));

if ~exist(testFolder, 'dir')
    mkdir(testFolder);
end
if ~exist(validationFolder, 'dir')
    mkdir(validationFolder);
end

%% Rutas de datos fuente
testPath = 'F:\RRSGAN-main\RRSGAN-main\dataset\val\val_1st\HR'; % Imágenes de prueba

%% Función para agregar ruido
add_noise = @(image, noise_dB) imnoise(image, 'gaussian', 0, 10^(-noise_dB / 10));

%% Procesar imágenes de prueba
disp('Procesando imágenes de prueba...');
testFiles = dir(fullfile(testPath, '*.jpg')); % Asumiendo formato JPG

for index = 1:length(testFiles)
    % Cargar imagen
    imgName = testFiles(index).name;
    imgPath = fullfile(testPath, imgName);
    img = imread(imgPath);

    % Normalizar imagen
    imgDouble = double(img);
    imgNorm = imgDouble / max(imgDouble(:)); % Normalizar entre 0 y 1

    % Recortar y redimensionar para imágenes HR y LR
    HR = modcrop(imgNorm, upscale_factor); % Recorte para divisibilidad
    LR = imresize(HR, 1 / upscale_factor); % Imagen de baja resolución

    % Agregar ruido a la imagen LR
    LR_noisy = add_noise(LR, noise_dB);

    % Guardar en la carpeta de Prueba
    [~, name, ~] = fileparts(imgName);
    save(fullfile(testFolder, [name, '.mat']), 'HR', 'LR', 'LR_noisy');
end

disp('Procesamiento completo.');


