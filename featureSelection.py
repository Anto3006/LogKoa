import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import re
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import copy
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import ParameterGrid
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import shap


ARCHIVO_CROSS_VALIDATION = "cross_validation.csv"

def verificarConstancia(datos, threshold = 0.8):
    columnasConstantes = []
    for columna in datos:
        conteo = {}
        total = 0
        for valor in datos[columna]:
            total += 1
            if valor in conteo:
                conteo[valor] += 1
            else:
                conteo[valor] = 1
        for valor in conteo:
            conteo[valor] /= total
        for valor in conteo:
            if conteo[valor] >= threshold:
                columnasConstantes.append(columna)
    return columnasConstantes
        

def scaleData(scaler,dataFrame):
    scaler = preprocessing.StandardScaler()
    scaler.fit(dataFrame)
    dataFrameScaled = pd.DataFrame(scaler.transform(dataFrame),columns=dataFrame.columns,index=dataFrame.index)
    return dataFrameScaled

def procesarDatos(datos,scale=False):
    y = datos["log_KOA"]
    x = datos.copy(deep=True)
    x.drop(columns=["log_KOA","id", "Cas_No","smiles"],inplace=True)
    x.dropna(axis=1,how="any",inplace=True)
    if scale:
        scaler = preprocessing.StandardScaler()
        x = scaleData(scaler,x)
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=3006)
    #Eliminar datos constantes
    columnasConstantes = verificarConstancia(x_train)
    x_train.drop(columns=columnasConstantes,inplace=True)
    x_test.drop(columns=columnasConstantes,inplace=True)
    #Eliminar datos altamentes correlacionados
    cor_matrix = x_train.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool_))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    x_train.drop(columns=to_drop,inplace=True)
    x_test.drop(columns=to_drop,inplace=True)
    return x_train,y_train,x_test,y_test





evaluacion = {"Modelo":[],"Feature Selection":[],"Train/Test":[],"R2":[],"RMSE":[]}
evaluacion_scaled = {"Modelo":[],"Feature Selection":[],"Train/Test":[],"R2":[],"RMSE":[]}

def entrenarModelo(modelo,x_train,y_train):
    modelo.fit(x_train,y_train)
    return modelo

def evaluarModelo(modelo,dic_evaluacion,x,y,nombreBase="figura",train=True,tipo="test",generarFigura=True,guardarValoresPredichos=True):
    r2 = modelo.score(x,y)
    dic_evaluacion["R2"].append(r2)
    adicional = "train"
    if not train:
        adicional = tipo
    y_pred = modelo.predict(x)
    diff = np.abs(y.to_numpy()-y_pred)
    valoresPredichos = pd.DataFrame()
    valoresPredichos["y"] = y
    valoresPredichos["y_pred"] = y_pred
    valoresPredichos.to_csv("valoresPredichos_" + adicional + "_" + nombreBase + ".csv")
    pd.DataFrame(data={"Diferencia":diff},index=x.index).to_csv("diff_" + adicional +"_"+nombreBase+".csv")
    if guardarValoresPredichos:
        np.savetxt("y_" + adicional +"_pred_"+nombreBase+".csv",y_pred,delimiter=",",header="log_KOA")
        x.to_csv("x_" + adicional +"_"+nombreBase+".csv")
    rmse = mean_squared_error(y, y_pred, squared=False)
    dic_evaluacion["RMSE"].append(rmse)
    #Marcar outlier del test
    if generarFigura:
        plt.plot(y_pred,y,'ro')
        plt.axline([0, 0], slope=1)
        plt.text(0,10,"R2="+str(round(r2,2)))
        plt.text(0,9,"RMSE="+str(round(rmse,2)))
        plt.savefig(nombreBase+"_"+adicional+".png")
        plt.clf()

#Genera los graficos SHAP para un modelo
def generarShap(modelo, nombreBase, x_train, x_test):
    explainer = shap.KernelExplainer(modelo.predict, x_train.iloc[0:100], keep_index=True)
    shap_valores = explainer.shap_values(x_test.iloc[0:50])
    shap.summary_plot(shap_valores, x_test.iloc[0:50],show=False)
    plt.savefig("shap"+nombreBase+".png")
    plt.clf()
    shap.summary_plot(shap_valores, x_test.iloc[0:50],show=False,plot_type="bar")
    plt.savefig("shap_bar"+nombreBase+".png")
    plt.clf()
    

def selectKBest(funcion_score,x_train,y_train,features=10):
    bestfeatures = SelectKBest(score_func=funcion_score, k=features)
    bestfeatures.fit(x_train,y_train)
    return bestfeatures


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





def hyperparametrosCV(modelo,x_train,y_train):
    scores = cross_val_score(modelo,x_train,y_train,cv=5,scoring="neg_root_mean_squared_error",error_score="raise")
    mean_score = np.mean(scores)
    return mean_score

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

def busquedaCompleta(modelo,nombreModelo,x_train,y_train,diccionarioHyperparametros,n_features="best"):
    
    """
    archivo = open(ARCHIVO_CROSS_VALIDATION,"a")
    funciones_score = {"f_regression":f_regression,"r_regression":r_regression,"mutual_info_regression":mutual_info_regression}
    for funcion in funciones_score:
        nombreArchivoResultadosTotales = nombreModelo+"_"+"k_best_"+funcion
        mejorResultado, mejoresHyper, features, numeroFeatures = gridSearch(modelo,"k_best",x_train,y_train,diccionarioHyperparametros,n_features,funcion_k_best=funciones_score[funcion],nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
        archivo.write(nombreModelo + ",k_best_" + funcion + "," + str(mejoresHyper).replace(',',' ') + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + str(features).replace(',',' ') + "\n")
    
    archivo.close()
    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"rfe"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"rfe",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    archivo = open(ARCHIVO_CROSS_VALIDATION,"a")
    archivo.write(nombreModelo + ",rfe" + "," + str(mejoresHyper).replace(',',' ') + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + str(features).replace(',',' ') + "\n")
    archivo.close()

    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"shap"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"shap",x_train,y_train,diccionarioHyperparametros,n_features,nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    archivo = open(ARCHIVO_CROSS_VALIDATION,"a")
    archivo.write(nombreModelo + ",shap" + "," + str(mejoresHyper).replace(',',' ') + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + str(features).replace(',',' ') + "\n")
    archivo.close()
    
    
    """
    nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_forward"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="forward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    archivo = open(ARCHIVO_CROSS_VALIDATION,"a")
    archivo.write(nombreModelo + ",sfs_forward" + "," + str(mejoresHyper).replace(',',' ') + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + str(features).replace(',',' ') + "\n")
    archivo.close()
    

    
    nombreArchivoResultadosTotales = nombreModelo+"_"+"sfs_backward"
    mejorResultado, mejoresHyper, features, numeroFeatures  = gridSearch(modelo,"sfs",x_train,y_train,diccionarioHyperparametros,n_features,direccion_sfs="backward",nombreArchivoResultadosTotales=nombreArchivoResultadosTotales)
    archivo = open(ARCHIVO_CROSS_VALIDATION,"a")
    archivo.write(nombreModelo + ",sfs_backward" + "," + str(mejoresHyper).replace(',',' ') + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + str(features).replace(',',' ') + "\n")
    archivo.close()
    

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
        hyperparametros = re.sub(r'\s+', ' ', hyperparametros).split(" ")
        dic_hyperparametros = {}
        for hyperparametro in hyperparametros:
            print(hyperparametro)
            datosHyper = hyperparametro.split("=")
            datosHyper[0].strip()
            nombreHyper = datosHyper[0]
            valorHyper = float(datosHyper[1])
            if nombreHyper in ["max_depth","n_estimators"]:
                valorHyper = int(valorHyper)
            dic_hyperparametros[nombreHyper] = valorHyper
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


