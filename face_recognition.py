from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

model = load_model('model/facenet_keras.h5')
print('Loaded Model')		
data = np.load('celeb_faces_dataset.npz', allow_pickle=True)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean() , face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	sample = np.expand_dims(face_pixels, axis=0)
	embedding = model.predict(sample)
	return embedding[0]


new_trainX = []
for face_pixels in trainX:
	embedding = get_embedding(model, face_pixels)	
	new_trainX.append(embedding)
new_trainX = np.asarray(new_trainX)

new_testX = []
for face_pixels in testX:
	embedding = get_embedding(model, face_pixels)	
	new_testX.append(embedding)
new_testX = np.asarray(new_testX)

in_encoder = Normalizer(norm='l2')
new_trainX = in_encoder.transform(new_trainX)
new_testX = in_encoder.transform(new_testX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

model = SVC(kernel="linear", probability=True)
model.fit(new_trainX, trainy)

yhat_train = model.predict(new_trainX)
yhat_test = model.predict(new_testX)

score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)

print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))