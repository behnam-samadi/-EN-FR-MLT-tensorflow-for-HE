import os
import cv2 as cv
import numpy as np
dataset_adress = "/share/users/bsamadi/datasets/image/val2014"
all_files = os.listdir(dataset_adress)

#select a part of data
selected_files = all_files
#print(selected_files)



with open("list_of_selected_files.txt", 'w') as f:
    for item in selected_files:
        f.writelines(item + "\n")
hist_dataset = np.ndarray((100, 256))
for i in range(100):
    item = selected_files[i]
    img = cv.imread(dataset_adress + "/"  + item)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([img_gray], [0], None, [256], [0,255])
    hist = hist[:,0]
    #hist = hist.reshape(-1)
    hist_dataset[i, :] = hist
    print(float(i)/len(selected_files))


