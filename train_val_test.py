import os
import shutil

image_folder = 'dataset\\images\\'
label_folder = 'dataset\\train\\'

image_files = [f for f in os.listdir('images')]
label_files = [f for f in os.listdir('labels')]

train_size = int(len(image_files) * 0.8)
val_size = int(len(image_files) * 0.1)

train = image_files[:train_size]
val = image_files[train_size:train_size+val_size]
test = image_files[train_size+val_size:]

for index, image in enumerate(train):
    label = label_files[index]
    shutil.move(os.path.join('images', image), os.path.join(image_folder + 'train', image))
    shutil.move(os.path.join('labels', label), os.path.join(label_folder + 'train', label))

for index, image in enumerate(val):
    label = label_files[train_size+index]
    shutil.move(os.path.join('images', image), os.path.join(image_folder + 'val', image))
    shutil.move(os.path.join('labels', label), os.path.join(label_folder + 'val', label))

for index, image in enumerate(test):
    label = label_files[train_size+val_size+index]
    shutil.move(os.path.join('images', image), os.path.join(image_folder + 'test', image))
    shutil.move(os.path.join('labels', label), os.path.join(label_folder + 'test', label))