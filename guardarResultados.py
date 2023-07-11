import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap

def mean_signed_error(y,y_pred):
    return ((y-y_pred)/len(y)).mean()

def evaluarModelo(modelo,x,y,nombreBase="figura",tipo="test",generarFigura=True):
    evaluacion = {}
    y_pred = modelo.predict(x)
    r2 = r2_score(y,y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mue = mean_absolute_error(y,y_pred)
    mse = mean_signed_error(y,y_pred)
    evaluacion["R2"] = r2
    evaluacion["RMSE"] = rmse
    evaluacion["MUE"] = mue
    evaluacion["MSE"] = mse
    valoresPredichos = pd.DataFrame()
    valoresPredichos["y"] = y
    valoresPredichos["y_pred"] = y_pred
    valoresPredichos.to_csv("valoresPredichos_" + tipo + "_" + nombreBase + ".csv")
    #Marcar outlier del test
    if generarFigura:
        plt.plot(y_pred,y,'ro')
        plt.axline([0, 0], slope=1)
        plt.text(0,10,"R2="+str(round(r2,2)))
        plt.text(0,9,"RMSE="+str(round(rmse,2)))
        plt.savefig(nombreBase+"_"+tipo+".png")
        plt.clf()
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
 