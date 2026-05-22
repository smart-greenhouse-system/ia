# from ultralytics import YOLO

# # ==========================================
# # CARGAR MODELO
# # ==========================================
# model = YOLO(
#     "/home/pablo/Documents/Python/crecimiento/tomate_ai/runs/segment/runs/segment/tomato_growth/weights/best.pt"
# )

# # ==========================================
# # VERIFICAR TASK
# # ==========================================
# print("TASK:", model.task)

# # ==========================================
# # IMAGEN DE PRUEBA
# # ==========================================
# image_path = (
#     "/home/pablo/Documents/Python/"
#     "crecimiento/tomate_ai/image/tomatoe.jpg"
# )

# # ==========================================
# # INFERENCIA
# # ==========================================
# results = model(
#     image_path,
#     conf=0.25
# )[0]

# # ==========================================
# # DEBUG
# # ==========================================
# print("BOXES:", results.boxes)

# print("MASKS:", results.masks)

# # ==========================================
# # MOSTRAR RESULTADO VISUAL
# # ==========================================
# results.show()

# # ==========================================
# # GUARDAR RESULTADO
# # ==========================================
# results.save(
#     filename="/home/pablo/Documents/Python/crecimiento/tomate_ai/image/result.jpg"
# )

# print("Imagen guardada.")


# from ultralytics import YOLO

# model = YOLO("models/detector_best.pt")

# print(model.task)

from ultralytics import YOLO

model_path = "/home/pablo/Documents/Python/crecimiento/tomate_ai/runs/segment/tomato_growth/weights/best.pt"

model = YOLO(model_path)

print(model.names)