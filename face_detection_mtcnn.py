import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from numpy import asarray, savez_compressed, load
from mtcnn.mtcnn import MTCNN
from glob import glob
import os

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = detector.detect_faces(pixels)
    faces = []
    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        faces.append(face_array)
    return faces

train_dir = 'data/train/'
X = []
y = []
for folder in os.listdir(train_dir):
    for file in glob(train_dir+folder+"/*"):
        face = extract_face(file)
        X.append(face[0])
        y.append(folder)


test_dir = 'data/val/'
X_val = []
y_val = []
for folder in os.listdir(test_dir):
    for file in glob(test_dir+folder+"/*"):
        face = extract_face(file)
        X_val.append(face[0])
        y_val.append(folder)

savez_compressed('celeb_faces_dataset.npz',X,y,X_val,y_val)

