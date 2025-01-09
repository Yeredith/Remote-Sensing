clear;
close all;
clc;

%% Definir hiperparámetros
patchSize = 32;            % Tamaño del parche
randomNumber = 10;         % Número de parches aleatorios
upscale_factor = 4;        % Factor de aumento
data_type = 'RS/normal';          % Nombre del tipo de datos
imagePatch = patchSize * upscale_factor;
scales = [1.0, 0.75, 0.5]; % Escalas de reducción

%% Crear carpeta de salida
currentFolder = pwd;
dataFolder = fullfile(currentFolder, 'Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end
trainFolder = fullfile(dataFolder, 'Validation');
if ~exist(trainFolder, 'dir')
    mkdir(trainFolder);
end
rsFolder = fullfile(trainFolder, data_type);
if ~exist(rsFolder, 'dir')
    mkdir(rsFolder);
end
scaleFolder = fullfile(rsFolder, num2str(upscale_factor), '\');
if ~exist(scaleFolder, 'dir')
    mkdir(scaleFolder);
end
savePath = scaleFolder;

%% Ruta de datos fuente
srPath = 'F:/SR-RS/train_data/train_data/HR';  % Ruta de las imágenes RGB
srFile = fullfile(srPath, '/');
srdirOutput = dir(fullfile(srFile, '*.jpg'));  % Asumiendo que las imágenes son .png
srfileNames = {srdirOutput.name}';
number = length(srfileNames);  % Número total de imágenes en la carpeta

%% Loop a través de todas las imágenes
for index = 3237:4046
    name = char(srfileNames(index));  % Obtener el nombre de la imagen
    if (isequal(name, '.') || isequal(name, '..'))
        continue;
    end
    disp(['----:', data_type, '----upscale_factor:', num2str(upscale_factor), '----procesando:', num2str(index), '----nombre:', name]);

    % Cargar la imagen RGB
    srImage = imread(fullfile(srPath, name));

    % Obtener las dimensiones de la imagen
    [height, width, Band] = size(srImage);

    % Verificar que la imagen es RGB (tres canales)
    if Band ~= 3
        warning(['La imagen ', name, ' no es RGB. Saltando...']);
        continue;
    end

    %% Normalización de la imagen utilizando el valor máximo
    imgz = double(srImage);             % Convertir la imagen a tipo double
    max_val = max(imgz(:));             % Obtener el valor máximo en la imagen
    img = imgz / max_val;               % Normalizar entre 0 y 1 dividiendo entre el valor máximo
    t = img;

    %% Procesar las escalas
    for sc = 1:length(scales)
        newt = imresize(t, scales(sc));  % Redimensionar la imagen
        x_random = randperm(size(newt, 1) - imagePatch, randomNumber);
        y_random = randperm(size(newt, 2) - imagePatch, randomNumber);

        for j = 1:randomNumber
            hrImage = newt(x_random(j):x_random(j) + imagePatch - 1, y_random(j):y_random(j) + imagePatch - 1, :);

            label = hrImage;   
            data_augment(label, upscale_factor, savePath);  % Guardar imagen original

            % Rotaciones y flip
            label = imrotate(hrImage, 180);  
            data_augment(label, upscale_factor, savePath);

            label = imrotate(hrImage, 90);
            data_augment(label, upscale_factor, savePath);

            % label = imrotate(hrImage, 270);
            % data_augment(label, upscale_factor, savePath);

            label = flipdim(hrImage, 1);
            data_augment(label, upscale_factor, savePath);

            label = flipdim(hrImage, 2);
            data_augment(label, upscale_factor, savePath);
        end
        clear x_random;
        clear y_random;
        clear newt;
    end
    clear t;
end

clear;
close all;
clc;

%% Definir hiperparámetros
patchSize = 32;            % Tamaño del parche
randomNumber = 10;         % Número de parches aleatorios
upscale_factor = 4;        % Factor de aumento
data_type = 'RS/noise';          % Nombre del tipo de datos
imagePatch = patchSize * upscale_factor;
scales = [1.0, 0.75, 0.5]; % Escalas de reducción
noise = 17;

%% Crear carpeta de salida
currentFolder = pwd;
dataFolder = fullfile(currentFolder, 'Data');
if ~exist(dataFolder, 'dir')
    mkdir(dataFolder);
end
trainFolder = fullfile(dataFolder, 'Validation');
if ~exist(trainFolder, 'dir')
    mkdir(trainFolder);
end
rsFolder = fullfile(trainFolder, data_type);
if ~exist(rsFolder, 'dir')
    mkdir(rsFolder);
end
scaleFolder = fullfile(rsFolder, num2str(upscale_factor), '\');
if ~exist(scaleFolder, 'dir')
    mkdir(scaleFolder);
end
savePath = scaleFolder;

%% Ruta de datos fuente
srPath = 'F:/SR-RS/train_data/train_data/HR';  % Ruta de las imágenes RGB
srFile = fullfile(srPath, '/');
srdirOutput = dir(fullfile(srFile, '*.jpg'));  % Asumiendo que las imágenes son .jpg
srfileNames = {srdirOutput.name}';
number = length(srfileNames);  % Número total de imágenes en la carpeta

%% Loop a través de todas las imágenes
for index = 3237:4046
    name = char(srfileNames(index));  % Obtener el nombre de la imagen
    if (isequal(name, '.') || isequal(name, '..'))
        continue;
    end
    disp(['----:', data_type, '----upscale_factor:', num2str(upscale_factor), '----procesando:', num2str(index), '----nombre:', name]);

    % Cargar la imagen RGB
    srImage = imread(fullfile(srPath, name));

    % Obtener las dimensiones de la imagen
    [height, width, Band] = size(srImage);

    % Verificar que la imagen es RGB (tres canales)
    if Band ~= 3
        warning(['La imagen ', name, ' no es RGB. Saltando...']);
        continue;
    end

    %% Normalización de la imagen utilizando el valor máximo
    imgz = double(srImage);             % Convertir la imagen a tipo double
    max_val = max(imgz(:));             % Obtener el valor máximo en la imagen
    img = imgz / max_val;               % Normalizar entre 0 y 1 dividiendo entre el valor máximo
    t = img;

    %% Procesar las escalas
    for sc = 1:length(scales)
        newt = imresize(t, scales(sc));

        % Generar posiciones aleatorias para los parches
        x_random = randi([1, size(newt, 1) - imagePatch + 1], [randomNumber, 1]);
        y_random = randi([1, size(newt, 2) - imagePatch + 1], [randomNumber, 1]);

        for j = 1:randomNumber
            % Extraer el parche de alta resolución (HR)
            hrImage = newt(x_random(j):x_random(j) + imagePatch - 1, y_random(j):y_random(j) + imagePatch - 1, :);

            % Añadir ruido con SNR de 17 dB a la imagen original
            noisyImage = addNoiseWithSNR(hrImage, noise);

            % Guardar imagen original con ruido
            data_augment(noisyImage, upscale_factor, savePath);

            % Rotaciones y flips
            label = imrotate(hrImage, 180);
            noisyLabel = addNoiseWithSNR(label, 17);
            data_augment(noisyLabel, upscale_factor, savePath);

            label = imrotate(hrImage, 90);
            noisyLabel = addNoiseWithSNR(label, 17);
            data_augment(noisyLabel, upscale_factor, savePath);

            % label = imrotate(hrImage, 270);
            % noisyLabel = addNoiseWithSNR(label, 17);
            % data_augment(noisyLabel, upscale_factor, savePath);

            label = flipdim(hrImage, 1);
            noisyLabel = addNoiseWithSNR(label, 17);
            data_augment(noisyLabel, upscale_factor, savePath);

            label = flipdim(hrImage, 2);
            noisyLabel = addNoiseWithSNR(label, 17);
            data_augment(noisyLabel, upscale_factor, savePath);
        end
        clear x_random;
        clear y_random;
        clear newt;
    end
    clear t;
end

%% Función para agregar ruido con SNR específico
function noisyImage = addNoiseWithSNR(image, targetSNR)
    % Convertir la imagen a tipo double
    image = double(image);

    % Calcular la potencia de la señal original
    signalPower = mean(image(:).^2);

    % Calcular la potencia del ruido necesaria para el SNR deseado
    snrLinear = 10^(targetSNR / 10);
    noisePower = signalPower / snrLinear;

    % Generar ruido gaussiano con la potencia calculada
    noise = sqrt(noisePower) * randn(size(image));

    % Añadir el ruido a la imagen
    noisyImage = image + noise;
    noisyImage = max(min(noisyImage, 1), 0);
end
