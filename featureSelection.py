import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import shap
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
import copy

def hyperparametrosCV(modelo,x_train,y_train):
    scores = cross_val_score(modelo,x_train,y_train,cv=5,scoring="neg_root_mean_squared_error",error_score="raise")
    mean_score = np.mean(scores)
    return mean_score

#Devuelve el nombre de la característica con el menor valor absoluto shap, la que se considera de menor importancia
def caracteristicaMenosImportante(modelo,x_train):
    x_background,x_shap_values = train_test_split(x_train,test_size=0.05,random_state=3006)
    x_train_summary = pd.DataFrame(data=shap.kmeans(x_background,10).data,columns=x_train.columns)
    explainer = shap.KernelExplainer(modelo,x_train_summary, keep_index=True)
    valores_shap = explainer.shap_values(x_shap_values)
    importancias = []
    #Calcula los valores shap para cada característica
    for i in range(valores_shap.shape[1]):
        importancias.append(np.mean(np.abs(valores_shap[:, i])))
    
    
    importancias_caracteristicas = {fea: imp for imp, fea in zip(importancias, x_train.columns)}
    
    importancias_caracteristicas = [(k,v) for k, v in sorted(importancias_caracteristicas.items(), key=lambda item: item[1], reverse = True)]

    return importancias_caracteristicas[-1]

def guardarResultadosTotales(resultados,archivoResultadosTotales):
    archivo = open(archivoResultadosTotales,"a")
    for i in range(len(resultados)):
        archivo.write(str(i+1) + "," + str(resultados[i]) + "\n")
    archivo.close()

def bestUnivariateFeatureSelection(modelo,x_train,y_train,funcionKBest=f_regression,k_features="best",archivoResultadosTotales=""):
    mejoresFeatures = []
    if k_features != "best":
        bestFeatures = SelectKBest(score_func=funcionKBest, k=k_features)
        bestFeatures.fit(x_train,y_train)
        mejoresFeatures = bestFeatures.get_feature_names_out()
    else:
        mejorResultado = -np.inf
        totalFeatures = len(x_train.columns)
        resultados = []
        for numberFeatures in range(1,totalFeatures+1):
            bestFeatures = SelectKBest(score_func=funcionKBest, k=numberFeatures)
            bestFeatures.fit(x_train,y_train)
            features = bestFeatures.get_feature_names_out()
            x_train_2 = x_train[features]
            cvScore = hyperparametrosCV(modelo,x_train_2,y_train)
            resultados.append(str(abs(cvScore)) + "," + str(features).replace(',', ' ').replace('\n',' '))
            if cvScore > mejorResultado:
                mejorResultado = cvScore
                mejoresFeatures = copy.deepcopy(features)
        if archivoResultadosTotales != "":
            guardarResultadosTotales(resultados,archivoResultadosTotales)
    return mejoresFeatures

def bestRFE(modelo,x_train,y_train,k_features="best",archivoResultadosTotales=""):
    mejoresFeatures = []
    if k_features != "best":
        selector = RFE(modelo,n_features_to_select=k_features)
        selector.fit(x_train,y_train)
        mejoresFeatures = selector.get_feature_names_out()
    else:
        selector = RFECV(modelo,step=1,cv=5,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        if archivoResultadosTotales != "":
            resultados = np.abs(np.array(selector.cv_results_["mean_test_score"]))
            guardarResultadosTotales(resultados,archivoResultadosTotales)
        mejoresFeatures = selector.get_feature_names_out()
    return mejoresFeatures

def bestSFS(modelo,x_train,y_train,k_features="best",direccionSFS="forward",archivoResultadosTotales=""):
    mejoresFeatures = []
    if k_features != "best":
        #8 jobs porque es demasiado pesado y se puede usar la computadora si se usan todos los procesadores
        selector = SequentialFeatureSelector(modelo,n_features_to_select=k_features,direction=direccionSFS,n_jobs=8,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        mejoresFeatures = selector.get_feature_names_out()
    else:
        print(direccionSFS)
        selector = SFS(modelo,k_features="best",forward=(direccionSFS=="forward"),n_jobs=8,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        mejoresFeatures = selector.k_feature_names_
        if archivoResultadosTotales != "":
            resultados = [str(selector.subsets_[i]['avg_score']) + "," + str(selector.subsets_[i]['feature_names']).replace(',',' ').replace('\n',' ') for i in range(1,len(list(selector.subsets_))+1)]
            guardarResultadosTotales(resultados,archivoResultadosTotales)
    return mejoresFeatures

def bestShap(modelo,x_train,y_train,k_features="best",archivoResultadosTotales=""):
    mejoresFeatures = []
    if k_features != "best":
        x_train_2 = x_train.copy(deep=True)
        modelo.fit(x_train_2,y_train)
        while len(x_train_2.columns) > k_features:
            print(len(x_train_2.columns))
            car = caracteristicaMenosImportante(modelo.predict,x_train_2)
            x_train_2.drop(columns=[car[0]],inplace=True)
            modelo.fit(x_train_2,y_train)
        mejoresFeatures = x_train_2.columns
    else:
        x_train_2 = x_train.copy(deep=True)
        modelo.fit(x_train_2,y_train)
        mejorResultado = hyperparametrosCV(modelo,x_train_2,y_train)
        mejoresFeatures = copy.deepcopy(x_train_2.columns)
        resultados = [str(mejorResultado) + "," + str(mejoresFeatures).replace(',',' ').replace('\n',' ')]
        while len(x_train_2.columns) > 1:
            print(len(x_train_2.columns))
            car = caracteristicaMenosImportante(modelo.predict,x_train_2)
            x_train_2.drop(columns=[car[0]],inplace=True)
            features = copy.deepcopy(x_train_2.columns)
            cvScore = hyperparametrosCV(modelo,x_train_2,y_train)
            modelo.fit(x_train_2,y_train)
            resultados.append(str(cvScore) + "," + str(features).replace(',',' ').replace('\n',' '))
            if cvScore > mejorResultado:
                mejorResultado = cvScore
                mejoresFeatures = copy.deepcopy(features)
        resultados.reverse()
        guardarResultadosTotales(resultados,archivoResultadosTotales)
    return mejoresFeatures

def featureSelection(modelo,tipo,x_train,y_train,k_features,funcion_k_best=f_regression,direccion_sfs="forward",nombreArchivoResultadosTotales=""):
    mejoresFeatures = []
    if tipo.lower() == "k_best":
        mejoresFeatures = bestUnivariateFeatureSelection(modelo,x_train,y_train,funcionKBest=funcion_k_best,k_features=k_features,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "rfe":
        mejoresFeatures = bestRFE(modelo,x_train,y_train,k_features=k_features,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "sfs":
        mejoresFeatures = bestSFS(modelo,x_train,y_train,k_features=k_features,direccionSFS=direccion_sfs,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "shap":
        mejoresFeatures = bestShap(modelo,x_train,y_train,k_features=k_features,archivoResultadosTotales=nombreArchivoResultadosTotales)
    return list(mejoresFeatures)