import cv2

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (640, 640))
    return image