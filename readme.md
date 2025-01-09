# Instrucciones
- Paso 1. Descargar el dataset de google drive. password:lnff
- Paso 2. Modificar el directorio de los archivos de la carpeta Preprocessing para generar los datos de entrenamiento, validacion y test con ruido y sin ruido y guardarlas de la siguiente forma.

- Remote Sensing

    -Data

        -Test

            -Noise
            
                -4

            - Normal

                -4
        -Train

            -Noise

                -4

            -Normal

                -4
        -Validation

            -Train
            
                - Noise

                - Normal


Ya los script generate_traindata.m y generate_Val_traindata.m tienen la division para datos de entrenamiento y validacion que se dividio en 80% entrenamiento y 20% Validacion y estos se ocupan unicamente con el dataset descargado en google drive y ocupar solo las imagenes de la carpeta HR (Verificar el directorio donde se encuentra guardada la carpeta).

Para el caso de ocupar generate_testdata.m se debe descargar la carpeta original_test_jpg y ocupar unicamente las imagenes de la carpeta HR (Verificar el directorio donde se encuentra guardada la carpeta).

Paso 3. Verificar en el json el directorio donde se pongan las carpetas.

Paso 4. Ejecutar main.
