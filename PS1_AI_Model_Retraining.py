#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow_hub')


# In[2]:


import numpy as np
import cv2

import PIL.Image as Image
import os

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])


# In[10]:


import subprocess
dataset_url = input()
f = subprocess.Popen(dataset_url)
data_dir = tf.keras.utils.get_file('pics', origin= f,  cache_dir='.', untar=True)


# In[ ]:


labels_dict = {}
m = int(input())
images_dict = [map(str,int(raw_input().split())) for j in range(m)]


# In[ ]:


images_dict = {}
n = int(input())
images_dict = [map(str,raw_input().split()) for i in range(n)]


# In[ ]:


X, y = [], []

for name, images in images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(224,224))
        X.append(resized_img)
        y.append(labels_dict[name])


# In[ ]:


X = np.array(X)
y = np.array(y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[ ]:


X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


# In[ ]:


feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# In[ ]:


model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(m)
])

model.summary()


# In[ ]:


model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)


# In[ ]:


model.evaluate(X_test_scaled,y_test)


# In[ ]:


scores = model.evaluate(X_test_scaled, y_test, verbose=0)

accuracy_file = open('/PS1_AI_Model_Retraining/show_accuracy.txt','w')
accuracy_file.write(str(scores[1]))
accuracy_file.close()

display_matter = open('/PS1_AI_Model_Retraining/show_output.html','r+')
display_matter.read()
display_matter.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
display_matter.close()
subprocess.Popen.terminate()

