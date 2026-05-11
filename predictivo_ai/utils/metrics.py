from sklearn.metrics import classification_report, confusion_matrix


def report(modelo, X_test, y_test, target_names):
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    matriz = confusion_matrix(y_test, y_pred)
    return reporte, matriz
