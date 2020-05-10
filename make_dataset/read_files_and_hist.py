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
hist_dataset = np.ndarray((len(selected_files), 256))
hist_dataset_equalized = np.ndarray((len(selected_files), 256))

for i in range(len(selected_files)):
    item = selected_files[i]
    img = cv.imread(dataset_adress + "/"  + item)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_gray_equalized = cv.equalizeHist(img_gray)
    hist = cv.calcHist([img_gray], [0], None, [256], [0,256])
    hist_eq = cv.calcHist([image_gray_equalized], [0], None, [256], [0,256])
    hist = hist[:,0]
    hist_eq = hist_eq[:,0]
    #hist = hist.reshape(-1)
    hist_dataset[i, :] = hist
    print(float(i)/len(selected_files))

raise(False)
np.save("input_histograms.npy", hist_dataset)
np.save("output_histograms.npy", hist_dataset_equalized)

