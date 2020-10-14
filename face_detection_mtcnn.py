import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from numpy import asarray, savez_compressed, load
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from glob import glob
import os

detector = MTCNN()

def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
   
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding boxes
    faces = []
    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        plt.imshow(image)
        plt.show()
        face_array = asarray(image)
        faces.append(face_array)
    return faces
 
# load the photo and extract the face
faces_array = extract_face('download.jpeg')


# In[2]:


train_dir = 'data/train/'
c = 0
X = []
y = []
for folder in os.listdir(train_dir):
    for file in glob(train_dir+folder+"/*"):
        face = extract_face(file)
        if(len(face) == 0):
            c = c + 1
        X.append(face[0])
        y.append(folder)
print(c)


# In[3]:


test_dir = 'data/val/'
c = 0
X_val = []
y_val = []
for folder in os.listdir(test_dir):
    for file in glob(test_dir+folder+"/*"):
        face = extract_face(file)
        if(len(face) == 0):
            c = c + 1
        X_val.append(face[0])
        y_val.append(folder)
print(c)


# In[4]:


savez_compressed('celeb_faces_dataset.npz',X,y,X_val,y_val)


# In[5]:


from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

def box_face(filename):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
   
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding boxes
    faces = []
    plt.imshow(image)
    ax = plt.gca()
    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        #image = Image.fromarray(face)
        #image = image.resize(required_size)
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)
        #plt.imshow(image)
        #plt.show()
        face_array = asarray(image)
        faces.append(face_array)
    plt.show()
    
box_face('download.jpeg')


# In[ ]:




