import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

def mean_signed_error(y,y_pred):
    return ((y-y_pred)/len(y)).mean()

def max_positive_error(y,y_pred):
    return (y-y_pred).max()

def max_negative_error(y,y_pred):
    return (y-y_pred).min()

def realizarEvaluaciones(y,y_pred):
    evaluacion = {}
    r2 = r2_score(y,y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mue = mean_absolute_error(y,y_pred)
    mse = mean_signed_error(y,y_pred)
    mpe = max_positive_error(y,y_pred)
    mne = max_negative_error(y,y_pred)
    evaluacion["R2"] = r2
    evaluacion["RMSE"] = rmse
    evaluacion["MUE"] = mue
    evaluacion["MSE"] = mse
    evaluacion["MPE"] = mpe
    evaluacion["MNE"] = mne
    return evaluacion

def generarFiguraEvaluacion(y,y_pred,evaluacion,nombreBase,tipo):
    plt.plot(y_pred,y,'ro')
    plt.axline([0, 0], slope=1)
    plt.text(0,10,"R2="+str(round(evaluacion["R2"],2)))
    plt.text(0,9,"RMSE="+str(round(evaluacion["RMSE"],2)))
    plt.savefig(nombreBase+"_"+tipo+".png")
    plt.clf()
    
def evaluarModelo(modelo,x,y,nombreBase="figura",tipo="test",generarFigura=True):
    y_pred = modelo.predict(x)
    evaluacion = realizarEvaluaciones(y,y_pred)
    valoresPredichos = pd.DataFrame()
    valoresPredichos["y"] = y
    valoresPredichos["y_pred"] = y_pred
    valoresPredichos.to_csv("valoresPredichos_" + tipo + "_" + nombreBase + ".csv")
    if generarFigura:
        generarFiguraEvaluacion(y,y_pred,evaluacion,nombreBase,tipo)
    return evaluacion

#Genera los graficos SHAP para un modelo
def generarShap(modelo, nombreBase, x_train, x_test):
    explainer = shap.KernelExplainer(modelo.predict, x_train, keep_index=True)
    shap_valores = explainer.shap_values(x_test)
    shap.summary_plot(shap_valores, x_test,show=False)
    plt.savefig("shap"+nombreBase+".png")
    plt.clf()
    shap.summary_plot(shap_valores, x_test,show=False,plot_type="bar")
    plt.savefig("shap_bar"+nombreBase+".png")
    plt.clf()

def guardarResultadosBusqueda(nombreArchivo,nombreModelo,featureSelection,mejorResultado,mejoresHyper,features,numeroFeatures):
    mejoresHyper = str(mejoresHyper).replace('{','').replace('}','').replace(':','=').replace("'","").replace(' ','')
    features = str(features).replace('[','').replace(']','').replace(',','').replace("'","")
    archivo = open(nombreArchivo,"a")
    archivo.write(nombreModelo + "," + featureSelection + "," + mejoresHyper + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + features + "\n")
    archivo.close()
 