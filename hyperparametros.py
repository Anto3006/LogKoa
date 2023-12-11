from sklearn.model_selection import ParameterGrid
from featureSelection import hyperparametrosCV
import copy
from sklearn.feature_selection import f_regression
import numpy as np

def gridSearch(modelo,featureSelectionMethod,x_train,y_train,diccionarioHyperparametros):
    listaHyper = list(ParameterGrid(diccionarioHyperparametros))
    mejorResultado = -np.inf
    mejoresHyper = []
    features = []
    mejoresFeatures = []
    for hyper in listaHyper:
        modelo.set_params(**hyper)
        print(hyper)
        featureSelectionMethod.hyperparameters = ""
        for h in hyper:
            featureSelectionMethod.hyperparameters += h + "_" + str(hyper[h])
        featureSelectionMethod.selectFeatures(modelo,x_train,y_train)
        features = featureSelectionMethod.bestFeatures
        resultado = featureSelectionMethod.bestScore
        f = open("a.txt",'a')
        f.write(str(resultado) + " "  + str(hyper) + " " + str(features) + "\n")
        f.close()
        if resultado > mejorResultado:
            mejorResultado = resultado
            mejoresHyper = copy.deepcopy(hyper)
            mejoresFeatures = copy.deepcopy(features)
    mejoresFeatures = list(mejoresFeatures)
    return abs(mejorResultado),mejoresHyper,mejoresFeatures,len(mejoresFeatures)

