import cv2

def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image

def to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
