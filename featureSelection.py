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
import copy
import time

ONLY_CPU = False
try:
    from cuml.explainer import KernelExplainer as cumlKernelExplainer
except ImportError:
    print("No se encuentran instalados los paquetes para usar los modelos en GPU, solo podra usarlos en CPU")
    ONLY_CPU = True



class FeatureSelectionMethod:

    def __init__(self,parameters):
        self.parameters = parameters
        self.bestFeatures = []
        self.hyperparameters = ""
    
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
        self.scoreFunctions = {"f_regression":f_regression,"r_regression":r_regression,"mutual_info_regression":mutual_info_regression}
    
    def selectBestFeaturesUnlimited(self,model,x_train,y_train):
        bestResult = -np.inf
        totalFeatures = len(x_train.columns)
        results = []
        featureSelector = SelectKBest(score_func=self.scoreFunctions[self.parameters["scoreFunc"]], k=totalFeatures)
        featureSelector.fit(x_train,y_train)
        bestFeaturesSortedIndex = sorted([index for index in range(totalFeatures)],key= lambda index: featureSelector.scores_[index], reverse=True)
        bestFeaturesSorted = featureSelector.feature_names_in_[bestFeaturesSortedIndex]
        beginTimeInterval = time.time()
        for numberFeatures in range(1,totalFeatures+1):
            features = bestFeaturesSorted[0:numberFeatures]
            x_train_2 = x_train[features]
            cvScore = hyperparametrosCV(model,x_train_2,y_train)
            results.append(str(abs(cvScore)) + "," + str(features).replace(',', ' ').replace('\n',' '))
            if cvScore > bestResult:
                bestResult = cvScore
                self.bestFeatures = copy.deepcopy(features)
            endTimeInterval = time.time()
            timePassed = endTimeInterval - beginTimeInterval
            if timePassed > 300:
                if self.parameters["fileAllResults"] != "":
                    guardarResultadosTotales(results,self.parameters["fileAllResults"]+self.hyperparameters)
                results = []
                beginTimeInterval = endTimeInterval
        if self.parameters["fileAllResults"] != "":
            guardarResultadosTotales(results,self.parameters["fileAllResults"]+self.hyperparameters)

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
        if self.parameters["fileAllResults"] != "":
            results = np.abs(np.array(selector.cv_results_["mean_test_score"]))
            guardarResultadosTotales(results,self.parameters["fileAllResults"]+self.hyperparameters)
        self.bestFeatures = selector.get_feature_names_out()
    
    def selectBestFeaturesK(self, model, x_train, y_train):
        selector = RFE(model,n_features_to_select=self.parameters["number_features"])
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
        if self.parameters["fileAllResults"] != "":
            results = [str(selector.subsets_[i]['avg_score']) + "," + str(selector.subsets_[i]['feature_names']).replace(',',' ').replace('\n',' ') for i in range(1,len(list(selector.subsets_))+1)]
            guardarResultadosTotales(results,self.parameters["fileAllResults"]+self.hyperparameters)

class RecursiveFeatureEliminationSHAP(FeatureSelectionMethod):

    def __init__(self, parameters):
        super().__init__(parameters)
    
    def selectBestFeaturesK(self, model, x_train, y_train):
        x_train_2 = x_train.copy(deep=True)
        model.fit(x_train_2,y_train)
        while len(x_train_2.columns) > self.parameters["number_features"]:
            print(len(x_train_2.columns))
            car = caracteristicaMenosImportante(model.predict,x_train_2,shap_split=self.parameters["background_split"],useGPU=self.parameters["gpu"])
            x_train_2.drop(columns=[car[0]],inplace=True)
            model.fit(x_train_2,y_train)
        self.bestFeatures = x_train_2.columns
    
    def selectBestFeaturesUnlimited(self, model, x_train, y_train):
        x_train_2 = x_train.copy(deep=True)
        model.fit(x_train_2,y_train)
        bestResult = hyperparametrosCV(model,x_train_2,y_train)
        bestFeatures = copy.deepcopy(x_train_2.columns)
        results = [str(bestResult) + "," + str(bestFeatures).replace(',',' ').replace('\n',' ')]
        while len(x_train_2.columns) > 1:
            print(len(x_train_2.columns))
            car = caracteristicaMenosImportante(model.predict,x_train_2,shap_split=self.parameters["background_split"],useGPU=self.parameters["gpu"])
            x_train_2.drop(columns=[car[0]],inplace=True)
            features = copy.deepcopy(x_train_2.columns)
            cvScore = hyperparametrosCV(model,x_train_2,y_train)
            model.fit(x_train_2,y_train)
            results.append(str(cvScore) + "," + str(features).replace(',',' ').replace('\n',' '))
            if cvScore > bestResult:
                bestResult = cvScore
                bestFeatures = copy.deepcopy(features)
        results.reverse()
        guardarResultadosTotales(results,self.parameters["fileAllResults"]+self.hyperparameters)
        self.bestFeatures = bestFeatures

def createFeatureSelectionMethod(featureSelectionName, parameters):
    if featureSelectionName == "UFS":
        return UnivariateFeatureSelection(parameters)
    elif featureSelectionName == "RFE":
        return RecursiveFeatureElimination(parameters)
    elif featureSelectionName == "RFE_SHAP":
        return RecursiveFeatureEliminationSHAP(parameters)
    elif featureSelectionName == "SFS":
        return SequentialFeatureSelection(parameters)


def hyperparametrosCV(modelo,x_train,y_train):
    scores = cross_val_score(modelo,x_train,y_train,cv=5,scoring="neg_root_mean_squared_error",error_score="raise")
    mean_score = np.mean(scores)
    return mean_score

#Devuelve el nombre de la característica con el menor valor absoluto shap, la que se considera de menor importancia
def caracteristicaMenosImportante(modelo,x_train,shap_split=0.05,useGPU=False):
    x_background,x_shap_values = train_test_split(x_train,test_size=shap_split,random_state=3006)
    x_train_summary = pd.DataFrame(data=shap.kmeans(x_background,10).data,columns=x_train.columns)
    explainer = None
    if ONLY_CPU or not useGPU:
        explainer = shap.KernelExplainer(modelo,x_train_summary, keep_index=True)
    else:
        explainer = cumlKernelExplainer(model=modelo,data=x_train_summary,random_state=3006)
    valores_shap = explainer.shap_values(x_shap_values)
    importancias = []
    #Calcula los valores shap para cada característica
    for i in range(valores_shap.shape[1]):
        importancias.append(np.mean(np.abs(valores_shap[:, i])))
    
    
    importancias_caracteristicas = {fea: imp for imp, fea in zip(importancias, x_train.columns)}
    
    importancias_caracteristicas = [(k,v) for k, v in sorted(importancias_caracteristicas.items(), key=lambda item: item[1], reverse = True)]

    return importancias_caracteristicas[-1]

def guardarResultadosTotales(resultados,archivoResultadosTotales):
    archivo = open(archivoResultadosTotales + ".csv","a")
    for i in range(len(resultados)):
        archivo.write(str(i+1) + "," + str(resultados[i]) + "\n")
    archivo.close()