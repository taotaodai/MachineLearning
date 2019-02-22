# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_lfw_people
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

print("{}".format(datasets.get_data_home()))
people = fetch_lfw_people(min_faces_per_person=0,resize=0.7)


iamge_shape = people.images[0].shape

fix,axes = plt.subplots(2,5,figsize=(15,8),subplot_kw={'xticks':(),'yticks':()})

for target,image,ax in zip(people.target,people.images,axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])

print("{}".format(people.images.shape))
print("{}".format(people.target_names))

#"显示每个人出现的次数"
#counts = np.bincount(people.target)
#
#for i,(count,name) in enumerate(zip(counts,people.target_names)):
#    print("{0:25} {1:3}".format(name,count),end=' ')
#    if(i+1) % 3 ==0:
#        print()

#mask = np.zeros(people.target.shape,dtype=np.bool)
#for target in np.unique(people.target):
#    mask[np.where(people.target = target)[0][:20]] = 1
#    X_people = people.data[mask]
#    y_people = people.target[mask]
#    
#    X_people = X_people / 255;