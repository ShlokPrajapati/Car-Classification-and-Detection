import os
import pandas as pd 
import matplotlib.pyplot as plt
root_path='car_data/train'
names=pd.read_csv('names.csv')
paths=[]
# print(names)
# for i in names:
#     paths=root_path+'/'+i
# print(paths)
data=[]
for car_folder in os.listdir(root_path):
    # car_folder_path = os.path.join(root_path, car_folder)
    car_folder_path = root_path+'/'+car_folder
    if os.path.isdir(car_folder_path):
        # Get all image files in the car folder
        images = [img for img in os.listdir(car_folder_path) if img.endswith('.jpg') or img.endswith('.png')]
        for image in os.listdir(car_folder_path):
            if (image.endswith('.jpg') or image.endswith('.png')):
                # Append data for the CSV
                data.append({'Car Name': car_folder, 'Images': image})
    # print(data)
    # print(car_folder_path)
df = pd.DataFrame(data)
print(df)
df.describe()
df.to_csv('all.csv')