import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap

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

def guardarResultadosBusqueda(nombreArchivo,nombreModelo,featureSelection,mejorResultado,mejoresHyper,features,numeroFeatures):
    mejoresHyper = str(mejoresHyper).replace('{','').replace('}','').replace(':','=').replace("'","").replace(' ','')
    features = str(features).replace('[','').replace(']','').replace(',','').replace("'","")
    archivo = open(nombreArchivo,"a")
    archivo.write(nombreModelo + "," + featureSelection + "," + mejoresHyper + "," + str(mejorResultado) + "," + str(numeroFeatures) + "," + features + "\n")
    archivo.close()
 