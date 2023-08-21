import pandas as pd
from procesarDatos import procesarHyperparametros
from modelos import crearModelo
import pickle
from lectorParametros import LectorParametros
import re

def entrenarMejoresModelos(nombreArchivoCrossValidation,nombreArchivoDatos,crossValidationThreshold):
    datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
    y_train = datos[datos.columns[1]]
    x_train = datos.drop(columns=["smiles",datos.columns[1]])
    archivoCV = pd.read_csv("CrossValidation/"+nombreArchivoCrossValidation)
    guardarModelosMejoresCV(archivoCV,x_train,y_train,crossValidationThreshold)

def guardarModelosMejoresCV(archivo,x_train,y_train,threshold):
    datos = archivo
    for index, row in datos.iterrows():
        if str(row["Score"]) != " " and float(row["Score"]) < threshold:
            hyp = str(row["Hyperparametros"])
            if hyp == "nan" or hyp == " ":
                hyp = ""
            nombreModelo = row["Modelo"].strip()
            print(nombreModelo)
            modelo = crearModelo(nombreModelo)
            if hyp != "":
                dicHyperparametros = procesarHyperparametros(hyp)
                modelo.set_params(**dicHyperparametros)
            features = re.sub(r'\s+', ' ', row["Features"])
            features = features.split(" ")
            features = [feature for feature in features if feature != ""]
            print(features)
            x_train_2 = x_train[features]
            modelo.fit(x_train_2,y_train)
            pickle.dump(modelo,open("Modelos/"+nombreModelo+"_"+row["Feature Selection"]+".sav",'wb'))

def main():
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    nombreArchivoCrossValidation = diccionarioValores["validacion"]
    nombreArchivoDatos = diccionarioValores["datos"]
    crossValidationThreshold = diccionarioValores["cvThreshold"]
    entrenarMejoresModelos(nombreArchivoCrossValidation,nombreArchivoDatos,crossValidationThreshold)

if __name__ == "__main__":
    main()