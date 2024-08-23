import csv
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms

def cutmix(image1, image2):
    file_name = [image1.filename, image2.filename]
    imgs = [image1.tobytes(), image2.tobytes()]


    rs = {} 
    for i in range(len(file_name)):
        rs[i] = file_name[i]

    name1 = rs[0]
    name2 = rs[1]
    print (rs)

    with open('img_name_label_testset.csv') as file_obj:
        reader_obj = csv.reader(file_obj)
        lable1 = None
        lable2 = None
        for row in reader_obj:
            if row[1] == name1 and lable1 is None:
                lable1 = row[3]
            if row[1] == name2 and lable2 is None:
                lable2 = row[3]
            if lable1 is not None and lable2 is not None:
                break

    rn= {}
    rn[0] = lable1
    rn[1] = lable2
    print (rn)

    img1 = imgs[0]
    img2 = imgs[1]

    image1 = Image.open(io.BytesIO(img1))
    image2 = Image.open(io.BytesIO(img2))

    img1_array = np.array(image1)
    img2_array = np.array(image2)
    print(img1_array.shape)

    img_exists = True

    if img_exists:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(254),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        pass

    sample_num = 2

    tensor1=test_transform(image1)
    tensor2=test_transform(image2)
    print(tensor1.shape)

    input_sp = torch.stack([tensor1, tensor2], dim=0)
    targets = torch.tensor([int(lable1), int(lable2)])

    alpha=0.5
    data=input_sp
    _, _, height, width = data.shape
    print(data.size(0))
    indices=[1,0]
    shuffled_data = data[indices]
    shuffled_targets =[]
    sorted_indexes = torch.argsort(targets, descending=True)
    shuffled_targets = torch.index_select(targets, index=sorted_indexes,dim=0,)

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform

from PIL import Image

image1 = Image.open("ILSVRC2012_val_00021015.JPEG")
image2 = Image.open("ILSVRC2012_val_00028673.JPEG")


result = cutmix(image1, image2)
