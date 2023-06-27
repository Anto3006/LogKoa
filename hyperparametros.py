from sklearn.model_selection import ParameterGrid
from featureSelection import featureSelection
from featureSelection import hyperparametrosCV
import copy
from sklearn.feature_selection import f_regression
import numpy as np

def gridSearch(modelo,tipo,x_train,y_train,diccionarioHyperparametros,n_features="best",funcion_k_best=f_regression,direccion_sfs="forward",nombreArchivoResultadosTotales=""):
    listaHyper = list(ParameterGrid(diccionarioHyperparametros))
    mejorResultado = -np.inf
    mejoresHyper = []
    features = []
    mejoresFeatures = []
    nombreGeneral = nombreArchivoResultadosTotales
    print(tipo)
    for hyper in listaHyper:
        modelo.set_params(**hyper)
        print(hyper)
        if nombreArchivoResultadosTotales != "":
            nombreArchivoResultadosTotales = nombreGeneral 
            for h in hyper:
                nombreArchivoResultadosTotales = nombreArchivoResultadosTotales + "_" + h + "_" + str(hyper[h])
        features = featureSelection(modelo,tipo,x_train,y_train,n_features,funcion_k_best,direccion_sfs,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales+".csv")
        x_train_2 = x_train[features]
        resultado = hyperparametrosCV(modelo,x_train_2,y_train)
        f = open("a.txt",'a')
        f.write(str(resultado) + " "  + str(hyper) + " " + str(features) + "\n")
        f.close()
        if resultado > mejorResultado:
            mejorResultado = resultado
            mejoresHyper = copy.deepcopy(hyper)
            mejoresFeatures = copy.deepcopy(features)
    mejoresFeatures = list(mejoresFeatures)
    return abs(mejorResultado),mejoresHyper,mejoresFeatures,len(mejoresFeatures)