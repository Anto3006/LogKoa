# Entrenar modelos de Machine Learning para quimioinformática

Esta herramienta permite a partir de SMILES y valores objetivo, buscar entre varios modelos de Machine Learning aplicando Feature Selection y optimización de hiperparámetros, para encontrar las combinaciones que den los mejores resultados.
El proceso se realiza en el orden siguiente:
- Cálculo automático de descriptores a partir del SMILES
- Preprocesamiento de los datos, eliminando descriptores constantes y datos faltantes
- Búsqueda entre los diferentes modelos, hiperparametros y métodos de feature selection
- Guardar los mejores modelos para su reutilización
- Evaluación de los modelos

## Cálculo de descriptores

- Agregar en la carpeta Datasets el archivo .csv con los SMILES y valores objetivo, los SMILES en la primera columna y el objetivo en la segunda
- Ejecutar el archivo calcularDescriptores.py seleccionando el archivo con los datos
- (Opcional) Se le puede dar una opcion de split con un valor entre 0 y 1 si se quiere dividir los datos en un set de training y otro de testing
```console
python calcularDescriptores.py -d archivoDatos.csv -s 0.15
```
- También se puede usar main.py, se tiene que usar con la opcion desc
```
python main.py desc -d archivoDatos.csv -s 0.15
```
- Los archivos con los descriptores calculados se crean en la carpeta Datasets

## Procesamiento de los datos

- Elimina columnas constantes, columnas o filas con datos faltantes y columnas altamente correlacionadas.
- El archivo con los datos tiene que estar en la carpeta Datasets, si lo obtuvo al calcular los descriptores con esta herramienta ya se encuentra ahí, nada más verifique el nombre del archivo.
- Ejecutar el archivo procesarDatos.py seleccionando el archivo con los datos
- (Opcional) Se puede cambiar el threshold para eliminar columnas con valores NA, con la opciones --colNA. Si una columna tiene una proporción mayor de colNA a la indicada se elimina.
- (Opcional) Se puede cambiar el threshold de constancia y de correlación, con las opciones --const y --corr respectivamente
```
python procesarDatos.py -d archivoDatos.csv --const 0.8 --corr 0.95 --colNA 0.2
```
- Puede hacer lo mismo con el archivo main.py, con la opción proc
```
python main.py proc -d archivoDatos.csv
```

## Búsqueda de modelos
- Prueba diferentes modelos: Regresión Lineal, Random Forest, SVM con kernel lineal y XGBoost, aplicandoles diferentes métodos de feature selection y haciendo una optimización de hiperparámetros con grid search para encontrar las mejores combinaciones
- En parametrosGridSearch.json se pueden seleccionar los parámetros para la búsqueda:
    - Qué modelos usar y si se entrenan con gpu o no
    - Los métodos de feature selection a usar y sus parámetros
    - La lista de hiperparámetros para realizar grid search
- Para realizar la búsqueda se ejecuta el archivo busqueda.py seleccionando el archivo con los datos y el nombre del archivo donde guardar los datos de cross-validation
```
python busqueda.py -d archivoDatos.csv -c archivoCrossValidation.csv
```
- Se puede realizar desde el archivo main con la opción search
```
python main.py search -d archivoDatos.csv -c archivoCrossValidation.csv
```

## Guardar los mejores modelos de acuerdo al cross-validation
- De acuerdo a su valor de cross-validation se seleccionan los mejores modelos, se entrenan y se guardan
- Se ejecuta el archivo guardarModelos.py seleccionando el archivo con los datos, el archivo con los cross-validation y el threshold de cross-validation score para guardar un modelo
```
python guardarModelos.py -d archivoDatos.csv -c archivoCrossValidation.csv -ct 0.5
```
- Se puede ejecutar desde el archivo main con la opción train
```
python main.py train -d archivoDatos.csv -c archivoCrossValidation.csv -ct 0.5
```

## Evaluar modelos
- Se evaluan los modelos con un set de datos de prueba y usando diferentes métricas
- Se ejecuta el archivo evaluar.py seleccionando el archivo con los datos de prueba y la etiqueta que se le quiera dar a estos datos (ej: test, train, external)
```
python evaluar.py -d archivoDatosEvaluar.csv -t test
```
- Se puede ejecutar desde el archivo main con la opción evaluar
```
python main.py evaluar -d archivoDatosEvaluar.csv -t test
```