import os
import cv2 as cv
dataset_adress = "/share/users/bsamadi/datasets/image/val2014"
all_files = os.listdir(dataset_adress)

#select a part of data
selected_files = all_files
#print(selected_files)



with open("list_of_selected_files.txt", 'w') as f:
    for item in selected_files:
        f.writelines(item + "\n")

for i in range(len(selected_files)):
    item = selected_files[i]
    img = cv.imread(dataset_adress + "/"  + item)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print(float(i)/len(selected_files))
