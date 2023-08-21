from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
ONLY_CPU = False
try:
    from cuml import LinearRegression as cumlLinearRegression
    from cuml.ensemble import RandomForestRegressor as cumlRandomForestRegressor
    from cuml.svm import LinearSVR as cumlLinearSVR
except ImportError:
    print("No se encuentran instalados los paquetes para usar los modelos en GPU, solo podra usarlos en CPU")
    ONLY_CPU = True

def crearModelo(nombreModelo,gpu=False):
    modelo = None
    if nombreModelo == "Linear Regression":
        if not gpu:
            modelo = LinearRegression()
        else:
            modelo = cumlLinearRegression(algorithm="svd-qr")
    elif nombreModelo == "Random Forest":
        if not gpu:
            modelo = RandomForestRegressor(random_state=3006)
        else:
            modelo = cumlRandomForestRegressor(accuracy_metric="mse",random_state=3006,n_streams=1)
    elif nombreModelo == "SVM":
        if not gpu:
            modelo = LinearSVR(random_state=3006,max_iter=10000)
        else:
            modelo = cumlLinearSVR(max_iter=10000)
    elif nombreModelo == "XGBoost":
        if not gpu:
            modelo = xgb.XGBRegressor(tree_method="exact",random_state=3006)
        else:
            modelo = xgb.XGBRegressor(tree_method="gpu_hist",random_state=3006)
    return modelo