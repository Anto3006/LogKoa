import pandas as pd
from procesarDatos import procesarDatos, procesarHyperparametros
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from guardarResultados import guardarResultadosBusqueda, evaluarModelo
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from hyperparametros import gridSearch
from featureSelection import hyperparametrosCV
import re


ARCHIVO_CROSS_VALIDATION = "cross_validation.csv"

evaluacion = {"Modelo":[],"Feature Selection":[],"Train/Test":[],"R2":[],"RMSE":[]}


def busquedaCompleta(modelo,nombreModelo,x_train,y_train,diccionarioHyperparametros,n_features="best"):
    
    
    funciones_score = {"f_regression":f_regression,"r_regression":r_regression,"mutual_info_regression":mutual_info_regression}
    for funcion in funciones_score:
        nombreArchivoResultadosTotales = nombreModelo+"_"+"k_best_"+funcion
        mejorResultado, mejoresHyper, features, numeroFeatures = gridSearch(modelo,"k_best",x_train,y_train,diccionarioHyperparametros,n_features,funcion_k_best=funciones_score[funcion],nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
        guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"k_best_"+funcion,mejorResultado,mejoresHyper,features,numeroFeatures)
    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"rfe"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"rfe",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"rfe",mejorResultado,mejoresHyper,features,numeroFeatures)

    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"shap"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"shap",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"shap",mejorResultado,mejoresHyper,features,numeroFeatures)
    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_forward"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="forward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"sfs_forward",mejorResultado,mejoresHyper,features,numeroFeatures)
    

    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_backward"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="backward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    guardarResultadosBusqueda(ARCHIVO_CROSS_VALIDATION,nombreModelo,"sfs_backward",mejorResultado,mejoresHyper,features,numeroFeatures)
    

def realizarPruebasMejoresCV(archivo,x_train,y_train,x_test,y_test,tipo="test"):
    datos = archivo
    evaluacion_local = evaluacion
    for index, row in datos.iterrows():
        if str(row["Score"]) != " " and float(row["Score"]) < 0.6:
            hyp = str(row["Hyperparametros"])
            if hyp == "nan" or hyp == " ":
                hyp = ""
            nombreModelo = row["Modelo"].strip()
            modelo = obtenerModelo(nombreModelo,hyp)
            features = re.sub(r'\s+', ' ', row["Features"])
            features = features.split(" ")
            features = [feature for feature in features if feature != ""]
            print(features)
            x_train_2 = x_train[features]
            x_test_2 = x_test[features]
            modelo.fit(x_train_2,y_train)
            print(row["Modelo"], row["Feature Selection"], hyperparametrosCV(modelo,x_train_2,y_train))
            evaluacion_local["Modelo"].append(row["Modelo"])
            evaluacion_local["Modelo"].append(row["Modelo"])
            evaluacion_local["Feature Selection"].append(row["Feature Selection"])
            evaluacion_local["Feature Selection"].append(row["Feature Selection"])
            evaluacion_local["Tipo"].append("Train")
            evaluarModelo(modelo,evaluacion_local,x_train_2,y_train,row["Modelo"]+"_"+row["Feature Selection"]+"_"+hyp,guardarValoresPredichos=True)
            evaluacion_local["Tipo"].append(tipo.capitalize())
            evaluarModelo(modelo,evaluacion_local,x_test_2,y_test,row["Modelo"]+"_"+row["Feature Selection"]+"_"+hyp,train=False,tipo=tipo,guardarValoresPredichos=True)
    pd.DataFrame(evaluacion).to_csv("evaluacion_external_total_no_limit.csv")

def obtenerModelo(nombreModelo,hyperparametros):
    modelo = None
    if nombreModelo == "Linear Regression":
        modelo = LinearRegression()
    elif nombreModelo == "Random Forest":
        modelo = RandomForestRegressor(random_state=3006)
    elif nombreModelo == "SVM":
        modelo = LinearSVR(random_state=3006,max_iter=10000)
    elif nombreModelo == "XGBoost":
        modelo = xgb.XGBRegressor(booster="gbtree",tree_method="exact",grow_policy="depthwise",random_state=3006)
    
    if hyperparametros != "":
        dic_hyperparametros = procesarHyperparametros(hyperparametros)
        modelo.set_params(**dic_hyperparametros)
    return modelo



datos = pd.read_csv("logKoa_descriptors_total.csv")

x_train,y_train,x_test,y_test = procesarDatos(datos,scale=False)

#x_train_scaled,_,x_test_scaled,_ = procesarDatos(datos,scale=True)

datosTest2 = pd.read_csv("externalCanon_descriptors_total_final.csv")
datosTest2.index = [475+i for i in range(len(datosTest2["No."]))]
y_external = datosTest2["log_Koa"]
x_external = datosTest2.drop(columns=["No.","Compound.name","log_Koa","smiles","CAS.RN"])


n_trees = [30,50,100,150,200,300,500,700,1000]
c_values = [0.001,0.002,0.005,0.007,0.01,0.02,0.05,0.07,0.1,0.2,0.5,0.7,1.0,2.0,5.0,10.0]
n_estimators = [100,500,1000]
learning_rate = [0.05,0.1,0.3]
max_depth = [5,10,20]

dic_xgboost = {"n_estimators":n_estimators,"learning_rate":learning_rate,"max_depth":max_depth}
dic_forest = {"n_estimators":n_trees}
dic_SVM = {"C":c_values}


modelo = LinearRegression()
nombreModelo = "Linear Regression"
busquedaCompleta(modelo,nombreModelo,x_train,y_train,{},n_features=10)


modelo = RandomForestRegressor(random_state=3006)
nombreModelo = "Random Forest"
busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_forest,n_features=10)


modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
nombreModelo = "XGBoost"
busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_xgboost,n_features=10)

modelo = LinearSVR(random_state=3006,max_iter=10000)
nombreModelo = "SVM"
busquedaCompleta(modelo,nombreModelo,x_train,y_train,dic_SVM,n_features=10)


"""
archivo = pd.read_csv("cross_validation_no_limit.csv")

realizarPruebasMejoresCV(archivo,x_train,y_train,x_external,y_external,tipo="external")
"""

