import shutil
import math
import os
import numpy

source_path = "D:\\code-projects\\KV6003-GAN\\datasets\\AHP"
target_path = "D:\\code-projects\\KV6003-GAN\\datasets\\AHP_Slice_v2"

train_size = 0.06
val_size = 0.06

def copy_files(source, destination, filelist):
    for i, filename in enumerate(filelist):
        src_annotation = os.path.join(source["Annotations"], filename + ".png") 
        src_image = os.path.join(source["JPEGImages"], filename + ".jpg")
        dst_annotation = os.path.join(destination["Annotations"], filename + ".png")
        dst_image = os.path.join(destination["JPEGImages"], filename + ".jpg")

        shutil.copyfile(src_annotation, dst_annotation)
        shutil.copyfile(src_image, dst_image)

        print(f"\rCopying files to {destination['root']}: {i+1}/{len(filelist)}" , end='')
    
    print()
    for i,file in enumerate(filelist):
        if not (os.path.exists(os.path.join(destination["Annotations"], file + ".png")) and os.path.exists(os.path.join(destination["JPEGImages"], file + ".jpg"))):
            raise Exception(f"File did not copy: {file}")
        print(f"\rChecking files in {destination['root']}: {i+1}/{len(filelist)}" , end='')
    print()

src = {
    "root": source_path,
    "JPEGImages": os.path.join(source_path,"train","JPEGImages"),
    "Annotations": os.path.join(source_path,"train","Annotations"),
    "ImageSets": os.path.join(source_path,"train","ImageSets")
}

train_dst = {
    "root": os.path.join(target_path,"train"),
    "JPEGImages": os.path.join(target_path,"train","JPEGImages"),
    "Annotations": os.path.join(target_path,"train","Annotations"),
    "ImageSets": os.path.join(target_path,"train","ImageSets")
}

val_dst = {
    "root": os.path.join(target_path,"val"),
    "JPEGImages": os.path.join(target_path,"val","JPEGImages"),
    "Annotations": os.path.join(target_path,"val","Annotations"),
    "ImageSets": os.path.join(target_path,"val","ImageSets")
}

for path in src.values():
    if not os.path.exists(path):
        raise Exception(f"Source folder {path} does not exist.")

for path in train_dst.values():
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder {path}")

for path in val_dst.values():
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder {path}")


with open(os.path.join(src["ImageSets"], "train.txt") , "r") as li:
    train_filelist = li.readlines()
    train_filelist = [file.strip('\n') for file in train_filelist]
    li.close()
    print("Read training file...")

with open(os.path.join(src["ImageSets"], "val.txt") , "r") as li:
    val_filelist = li.readlines()
    val_filelist = [file.strip('\n') for file in val_filelist]
    li.close()
    print("Read validation file...")

nTrain = math.ceil(train_size * len(train_filelist))
nVal = math.ceil(val_size * len(val_filelist))

numpy.random.shuffle(train_filelist)
numpy.random.shuffle(val_filelist)

train_filelist = train_filelist[0:nTrain]
val_filelist = val_filelist[0:nVal]

copy_files(src, train_dst, train_filelist)

with open(os.path.join(train_dst["ImageSets"], "train.txt") , "wt") as li:
    for file in train_filelist:
        li.write(file + "\n")
    li.close()

print("Finished copying training images")

copy_files(src, val_dst, val_filelist)
with open(os.path.join(val_dst["ImageSets"], "val.txt") , "wt") as li:
    for file in val_filelist:
        li.write(file + "\n")
    li.close()

print("Finished copying validation images")