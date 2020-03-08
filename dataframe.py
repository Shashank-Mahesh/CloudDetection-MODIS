#!/usr/bin/env python
# coding: utf-8

# In[25]:


import h5py
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

home = '/home/hackathon/output_64_Javier_labelled/'
save_path = '/home/smahesh3/ondemand/data/sys/dashboard/batch_connect/sys/jupyter-notebook/features_2.csv'


# In[26]:


all_files = []
all_features = []
all_classifications = []

errors = 0
no_files = 0
for file_name in os.listdir(home):
    if file_name.endswith('.hdf'):
        no_files += 1
        path = home + file_name
        try:
            file = h5py.File(path, 'r')
            
            files = []
            classifications = []
            features = []
            for key in list(file.keys()):
                accuracy = file[key + '/ClassificationAccuracy'][()]
                if accuracy == 1:
                    all_files.append(file)
                    all_features.append(file[key + '/ImageFeatures'][()])
                    all_classifications.append(file[key + '/ImageClassification'][()])
        except OSError:
            errors += 1


# In[27]:


images = len(all_features)
pixels = len(all_features[0][0]) * len(all_features[0][0][0])
array = np.empty([images * pixels, len(all_features[0])])




for i in range(len(all_features)):
    for j in range(len(all_features[i])):
        for k in range(len(all_features[i][j])):
            for l in range(len(all_features[i][j][k])):
                array[115 * i + 64 * k + l][j] = all_features[i][j][k][l]
df = pd.DataFrame(array)


# In[28]:


all_arrays = np.ones([64 * 64 * 115, 43])


# In[29]:


re = np.array(all_features).swapaxes(1, 3).swapaxes(2,1)
dfa = np.array(re).flatten().reshape(115*64*64, 43)
df = pd.DataFrame(dfa)


# In[30]:


dfl = []
type(all_features)

for i in range(64):
    for j in range(64):
        for k in range(43):
            array[64 * i + j][k] = all_features[1][k][i][j]


# In[31]:


class_array = np.empty([images * pixels])
for i in range(len(all_classifications)):
    for j in range(len(all_classifications[i])):
        for k in range(len(all_classifications[i][j])):
            class_array[i * j * k] = int(all_classifications[i][j][k])


# In[32]:


list_array = list(class_array)
df['Classification'] = class_array


# In[33]:


df.to_csv(save_path)


# In[24]:


# dropped_df = sample_df.replace(0, np.nan)
# dropped_df = dropped_df.dropna(how='all', axis=0)

