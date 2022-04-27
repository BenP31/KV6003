import os
import PIL.Image
import numpy as np

#Variables
target_dir = "D:/code-projects/KV6003-GAN/datasets/AHP_Slice_v2/val"

src = {
    "root": target_dir,
    "JPEGImages": os.path.join(target_dir,"JPEGImages"),
    "Annotations": os.path.join(target_dir,"Annotations"),
    "ImageSets": os.path.join(target_dir,"ImageSets")
}

dst = {
    "root": target_dir,
    "ObsImages": os.path.join(target_dir,"ObsImages"),
    "ObsAnnotations": os.path.join(target_dir,"ObsAnnotations"),
}

def remove_centre(arr, radius):
    assert radius < np.shape(arr)[1] or radius < np.shape(arr)[2]
    mid = np.shape(arr)[1]/2
    arr[:, mid-radius:mid+radius, mid-radius:mid+radius, :] = -1
    return arr

def remove_column(arr, centre_pos, width):
    left = centre_pos-(width/2)
    if left < 0: left = 0
    right = centre_pos+(width/2)
    if right > np.shape(arr)[2] : right = np.shape(arr)[2]
    arr[:, :, left:right, :] = -1
    return arr

def remove_row(arr, centre_pos, width):
    bottom = centre_pos-(width/2)
    if bottom < 0: left = 0
    top = centre_pos+(width/2)
    if top > np.shape(arr)[2] : top = np.shape(arr)[2] 
    arr[:, top:bottom, :, :] = -1
    return arr

def random_transform(inputs: list):
    random = np.random.random()
    transformed = []
    for x in inputs: 
        if random <= 0.66:
            trans_x = remove_centre(x, np.random.randint(5, 50))
        if random <= 0.33:
            trans_x = remove_column(x, np.random.randint(50, 450), np.random.randint(5,50))
        else:
            trans_x = remove_row(x, np.random.randint(50, 450), np.random.randint(5,50))
        transformed.append(trans_x)

    return transformed

if __name__ == "__main__":
    for path in dst.values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder {path}")

    for path in src.values():
        if not os.path.exists(path):
            raise Exception(f"Folder does not exist: {path}")

    with open(os.path.join(src["ImageSets"], "val.txt"), "r") as fp:
        filelist = fp.readlines()
        fp.close()
    filelist = [file.strip("\n") for file in filelist]

    for i, file in enumerate(filelist):
        with PIL.Image.open(os.path.join(src["Annotations"], file + ".png")) as anno, PIL.Image.open(os.path.join(src["JPEGImages"], file + ".jpg")) as jpegimg:
            anno_arr = np.array(anno)
            jpeg_arr = np.array(jpegimg)

        anno_arr, jpeg_arr = random_transform([anno_arr, jpeg_arr])

        np.save(os.path.join(dst["ObsAnnotations"],file+".npy"), anno_arr)
        np.save(os.path.join(dst["ObsImages"],file+".npy"), jpeg_arr)
        print(f"\rObscured {i+1}/{len(filelist)} images")
    print()
    print("Finished")