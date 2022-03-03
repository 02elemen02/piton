from time import sleep
import tensorflow
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import cv2

model = load_model('C:\\Users\\HP\\Downloads\\converted_keras2\\keras_model.h5',compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
camera = cv2.VideoCapture(0)# Video çekmeye başla
return_value,image = camera.read()# İlk fotğrafı al
cv2.imwrite('test.jpg',image)#Kaydet
camera.release()# ?
sleep(1)
image = Image.open('test.jpg')
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array
prediction = model.predict(data) 
dataFrame1 = pd.DataFrame(prediction)
head = dataFrame1.head() # ilk 5 elemanı verir ‘istege bağlı olarak head() fonk. içine istediğiniz kadarınıda alabilir.’
print("Uğur" ,head[0])
print("Sonay" ,head[1])
print("Aysel",head[2])
print("Fatma",head[3])




