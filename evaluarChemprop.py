from evaluar import realizarEvaluaciones
import os
import pandas as pd


evaluaciones = pd.DataFrame(columns=["R2","RMSE","MUE","MSE","MPE","MNE"])
y_total = pd.Series()
y_total_pred = pd.Series()
for nombreArchivo in os.listdir("chemprop"):
    archivo = pd.read_csv("chemprop/"+nombreArchivo)
    y = archivo["y"]
    y_pred = archivo["y_pred"]
    y_total = y_total.append(y)
    y_total_pred = y_total_pred.append(y_pred)
    resultados = realizarEvaluaciones(y,y_pred)
    evaluaciones = evaluaciones.append(resultados,ignore_index=True)
resultados = realizarEvaluaciones(y_total,y_total_pred)
evaluaciones = evaluaciones.append(resultados,ignore_index=True)
evaluaciones.index = os.listdir("chemprop") + ["total"]
evaluaciones.to_excel("evaluaciones_chemprop.xlsx")
