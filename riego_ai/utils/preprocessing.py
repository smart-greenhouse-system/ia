def validar_entrada(cantidad, threshold, caudal):
    if cantidad < 0:
        raise ValueError(f"cantidad ({cantidad}) no puede ser negativo")
    if threshold < 0:
        raise ValueError(f"threshold_minimo ({threshold}) no puede ser negativo")
    if caudal <= 0:
        raise ValueError(f"caudal_bomba_lps ({caudal}) debe ser mayor a 0")
    if cantidad < threshold:
        raise ValueError(
            f"cantidad ({cantidad}) es menor que threshold_minimo ({threshold}). "
            "El tanque ya esta en nivel minimo."
        )


def calcular_ratio(cantidad, threshold, caudal):
    volumen = max(0, cantidad - threshold)
    return volumen / caudal


def formatear_tiempo(segundos):
    if segundos is None or segundos < 0:
        return "0 segundos"
    minutos = int(segundos // 60)
    segs = int(segundos % 60)
    if minutos > 0:
        return f"{minutos} minutos y {segs} segundos"
    return f"{segs} segundos"
