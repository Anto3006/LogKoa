from urllib.parse import ParseResult
from symbol import parameters
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
def caracteristicaMenosImportante(modelo,x_train,shap_split=0.05):
    x_background,x_shap_values = train_test_split(x_train,test_size=shap_split,random_state=3006)
    x_train_summary = pd.DataFrame(data=shap.kmeans(x_background,10,n_init=10).data,columns=x_train.columns)
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


class FeatureSelectionMethod:

    def __init__(self,parameters):
        self.parameters = parameters
        self.bestFeatures = []
    
    def selectFeatures(self,model,x_train,y_train):
        if self.parameters["number_features"] == "best":
            self.selectBestFeaturesUnlimited(model,x_train,y_train)
        elif isinstance(self.parameters["number_features"],int):
            self.selectBestFeaturesK(model,x_train,y_train)
    
    def selectBestFeaturesUnlimited(self,model,x_train,y_train):
        pass

    def selectBestFeaturesK(self,model,x_train,y_train):
        pass

class UnivariateFeatureSelection(FeatureSelectionMethod):

    def __init__(self,parameters):
        super().__init__(parameters)
    
    def selectBestFeaturesUnlimited(self,model,x_train,y_train):
        bestResult = -np.inf
        totalFeatures = len(x_train.columns)
        results = []
        featureSelector = SelectKBest(score_func=self.parameters["scoreFunc"], k=totalFeatures)
        featureSelector.fit(x_train,y_train)
        bestFeaturesSortedIndex = sorted([index for index in range(totalFeatures)],key= lambda index: featureSelector.scores_[index], reverse=True)
        bestFeaturesSorted = featureSelector.feature_names_in_[bestFeaturesSortedIndex]
        for numberFeatures in range(1,totalFeatures+1):
            features = bestFeaturesSorted[0:numberFeatures]
            x_train_2 = x_train[features]
            cvScore = hyperparametrosCV(model,x_train_2,y_train)
            results.append(str(abs(cvScore)) + "," + str(features).replace(',', ' ').replace('\n',' '))
            if cvScore > bestResult:
                bestResult = cvScore
                self.bestFeatures = copy.deepcopy(features)
        if parameters["fileAllResults"] != "":
            guardarResultadosTotales(results,self.parameters["fileAllResults"])

    def selectBestFeaturesK(self, model, x_train, y_train):
        featureSelector = SelectKBest(score_func=self.parameters["scoreFunc"], k=self.parameters["number_features"])
        featureSelector.fit(x_train,y_train)
        self.bestFeatures = featureSelector.get_feature_names_out()

class RecursiveFeatureElimination(FeatureSelectionMethod):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def selectBestFeaturesUnlimited(self, model, x_train, y_train):
        selector = RFECV(model,step=1,cv=5,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        if parameters["fileAllResults"] != "":
            results = np.abs(np.array(selector.cv_results_["mean_test_score"]))
            guardarResultadosTotales(results,parameters["fileAllResults"])
        self.bestFeatures = selector.get_feature_names_out()
    
    def selectBestFeaturesK(self, model, x_train, y_train):
        selector = RFE(model,n_features_to_select=parameters["number_features"])
        selector.fit(x_train,y_train)
        self.bestFeatures = selector.get_feature_names_out()

class SequentialFeatureSelection(FeatureSelectionMethod):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def selectBestFeaturesK(self, model, x_train, y_train):
        selector = SequentialFeatureSelector(model,n_features_to_select=self.parameters["number_features"],direction=self.parameters["direction"],n_jobs=self.parameters["jobs"],scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        self.bestFeatures = selector.get_feature_names_out()
    
    def selectBestFeaturesUnlimited(self, model, x_train, y_train):
        selector = SFS(model,k_features="best",forward=(self.parameters["direction"]=="forward"),n_jobs=self.parameters["jobs"],scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        self.bestFeatures = selector.k_feature_names_
        if parameters["fileAllResults"] != "":
            results = [str(selector.subsets_[i]['avg_score']) + "," + str(selector.subsets_[i]['feature_names']).replace(',',' ').replace('\n',' ') for i in range(1,len(list(selector.subsets_))+1)]
            guardarResultadosTotales(results,parameters["fileAllResults"])

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
        bestFeatures = SelectKBest(score_func=funcionKBest, k=totalFeatures)
        bestFeatures.fit(x_train,y_train)
        bestFeaturesSortedIndex = sorted([index for index in range(totalFeatures)],key= lambda index: bestFeatures.scores_[index], reverse=True)
        bestFeaturesSorted = bestFeatures.feature_names_in_[bestFeaturesSortedIndex]
        for numberFeatures in range(1,totalFeatures+1):
            print(numberFeatures)
            features = bestFeaturesSorted[0:numberFeatures]
            x_train_2 = x_train[features]
            print("Inicio")
            cvScore = hyperparametrosCV(modelo,x_train_2,y_train)
            print("Final")
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
        selector = SequentialFeatureSelector(modelo,n_features_to_select=k_features,direction=direccionSFS,n_jobs=6,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        mejoresFeatures = selector.get_feature_names_out()
    else:
        selector = SFS(modelo,k_features="best",forward=(direccionSFS=="forward"),n_jobs=6,scoring="neg_root_mean_squared_error")
        selector.fit(x_train,y_train)
        mejoresFeatures = selector.k_feature_names_
        if archivoResultadosTotales != "":
            resultados = [str(selector.subsets_[i]['avg_score']) + "," + str(selector.subsets_[i]['feature_names']).replace(',',' ').replace('\n',' ') for i in range(1,len(list(selector.subsets_))+1)]
            guardarResultadosTotales(resultados,archivoResultadosTotales)
    return mejoresFeatures

def bestShap(modelo,x_train,y_train,k_features="best",shap_split=0.05,archivoResultadosTotales=""):
    mejoresFeatures = []
    if k_features != "best":
        x_train_2 = x_train.copy(deep=True)
        modelo.fit(x_train_2,y_train)
        while len(x_train_2.columns) > k_features:
            print(len(x_train_2.columns))
            car = caracteristicaMenosImportante(modelo.predict,x_train_2,shap_split=shap_split)
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
            car = caracteristicaMenosImportante(modelo.predict,x_train_2,shap_split=shap_split)
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

def featureSelection(modelo,tipo,x_train,y_train,k_features,funcion_k_best=f_regression,direccion_sfs="forward",shap_split=0.05,nombreArchivoResultadosTotales=""):
    mejoresFeatures = []
    if tipo.lower() == "k_best":
        mejoresFeatures = bestUnivariateFeatureSelection(modelo,x_train,y_train,funcionKBest=funcion_k_best,k_features=k_features,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "rfe":
        mejoresFeatures = bestRFE(modelo,x_train,y_train,k_features=k_features,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "sfs":
        mejoresFeatures = bestSFS(modelo,x_train,y_train,k_features=k_features,direccionSFS=direccion_sfs,archivoResultadosTotales=nombreArchivoResultadosTotales)
    elif tipo.lower() == "shap":
        mejoresFeatures = bestShap(modelo,x_train,y_train,k_features=k_features,shap_split=shap_split,archivoResultadosTotales=nombreArchivoResultadosTotales)
    return list(mejoresFeatures)