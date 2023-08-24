import sys
import getopt

class LectorParametros:

    def __init__(self):
        self.argumentList = sys.argv[1:]
        self.options = "d:c:t:s:h:"
        self.longOptions = ["dataset=","cross-validation=","tipo=","split=", "colNA=","const=", "corr=", "cvThreshold="]
    
    def leerParametros(self):
        parametros, valores = getopt.getopt(self.argumentList, self.options, self.longOptions)
        diccionarioValores = {}
        splitFlag = False
        porcentajeSplit = 0.15
        constThreshold = 0.8
        corrThreshold = 0.95
        colNAThreshold = 0
        for parametroActual, valorActual in  parametros:
            if parametroActual in ("-d","--dataset"):
                diccionarioValores["datos"] = valorActual
            elif parametroActual in ("-c","--cross-validation"):
                diccionarioValores["validacion"] = valorActual
            elif parametroActual in ("-t","--tipo"):
                diccionarioValores["tipo"] = valorActual
            elif parametroActual in ("-s","--split"):
                splitFlag = True
                porcentajeSplit = float(valorActual)
            elif parametroActual == "--const":
                diccionarioValores["constThreshold"] = float(valorActual)
            elif parametroActual == "--corr":
                diccionarioValores["corrThreshold"] = float(valorActual)
            elif parametroActual in ("-h","--cvThreshold"):
                diccionarioValores["cvThreshold"] = float(valorActual)
            elif parametroActual == "--colNA":
                colNAThreshold = float(valorActual)
        diccionarioValores["split"] = splitFlag
        diccionarioValores["porcentajeSplit"] = porcentajeSplit
        diccionarioValores["constThreshold"] = constThreshold
        diccionarioValores["corrThreshold"] = corrThreshold
        diccionarioValores["colNAThreshold"] = colNAThreshold
        return diccionarioValores

