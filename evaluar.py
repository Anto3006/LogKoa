import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
from lectorParametros import LectorParametros
import pickle
import os

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
def generarShap(modelo, nombreBase, x_train, x_test, titulo=""):
    explainer = shap.KernelExplainer(modelo.predict, x_train, keep_index=True)
    shap_valores = explainer.shap_values(x_test)
    shap.summary_plot(shap_valores, x_test,show=False)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig("shap"+nombreBase+".png",bbox_inches="tight")
    plt.clf()
    shap.summary_plot(shap_valores, x_test,show=False,plot_type="bar")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig("shap_bar"+nombreBase+".png",bbox_inches="tight")
    plt.clf()

def evaluarModelosGuardados(x,y,nombreArchivo,tipo="test",generarFigura=True):
    modelos = os.listdir('Modelos')
    modelos.sort()
    resultados = pd.DataFrame(columns=["R2","RMSE","MUE","MSE"])
    for nombreModelo in modelos:
        modelo = pickle.load(open("Modelos/"+nombreModelo, 'rb'))
        nombreModelo = ".".join(nombreModelo.split(".")[0:-1])
        features = modelo.feature_names_in_
        x_2 = x[features]
        evaluacion = evaluarModelo(modelo,x_2,y,nombreModelo,generarFigura=generarFigura)
        resultados = resultados.append(evaluacion,ignore_index=True)
    resultados.index = modelos
    modo = "w"
    if nombreArchivo+".xlsx" in os.listdir():
        modo = "a"
    writer = pd.ExcelWriter(nombreArchivo+".xlsx",engine="openpyxl",mode=modo)
    resultados.to_excel(writer,sheet_name=tipo,engine="openpyxl")
    writer.close()

def main():
    lector = LectorParametros()
    diccionarioValores = lector.leerParametros()
    nombreArchivoDatos = diccionarioValores["datos"]
    tipo = diccionarioValores["tipo"]
    datos = pd.read_csv("Datasets/"+nombreArchivoDatos)
    y = datos[datos.columns[1]]
    x = datos.drop(columns=[datos.columns[1],"smiles"])
    evaluarModelosGuardados(x,y,"evaluacion_modelos",tipo=tipo,generarFigura=False)

if __name__ == "__main__":
    main()
 