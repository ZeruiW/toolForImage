from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import csv
from PIL import Image
import io

'''AI library'''
# first you shuld install : pip install gco-wrapper
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms #transforms.ToTensor() changes image to tensor
import torchvision.models as models
import matplotlib.pyplot as plt
import math
import pickle
import mixup
from mixup import mixup_graph

os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%matplotlib inline

app = Flask(__name__)
api = Api(app) 

# define the print function
def print_fig(input, target=None, title=None, save_dir=None):
        fig, axes = plt.subplots(1,len(input),figsize=(3*len(input),3))
        if title:
            fig.suptitle(title, size=16)
        if len(input) == 1 :
            axes = [axes]

        for i, ax in enumerate(axes):
            if len(input.shape) == 4:
                ax.imshow(input[i].permute(1,2,0).numpy())
            else :
                ax.imshow(input[i].numpy(), cmap='gray', vmin=0., vmax=1.)

            if target is not None:
                output = net((input[i].unsqueeze(0) - mean)/std)
                loss = criterion(output, target[i:i+1])
                ax.set_title("loss: {:.3f}\n pred: {}\n true : {}".format(loss, CIFAR100_LABELS_LIST[output.max(1)[1][0]], CIFAR100_LABELS_LIST[target[i]]))
            ax.axis('off')
        plt.subplots_adjust(wspace = 0.1)

        if save_dir is not None:
            plt.savefig(save_dir, bbox_inches = 'tight',  pad_inches = 0)

        plt.show()


# Puzzlemix
@app.route('/Puzzlemix', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        files = request.files
        imgs = files.getlist('image')
        file_name = [img.filename for img in imgs]
        imgs = [i.read() for i in imgs] # this will read binary images

        rs = {} # it is a dict?
        for i in range(len(file_name)):
            rs[i] = file_name[i]

        name1 = rs[0]
        name2 = rs[1]
        print (rs) # {0: 'ILSVRC2012_val_00000324.JPEG', 1: 'ILSVRC2012_val_00000318.JPEG'}
        #return jsonify(rs)

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
        print (rn) # {0: '21', 1: '7'}
        #return jsonify(rn)


        for file_name in files:
            file = files[file_name]
            print(f"File Name: {file.filename}")
            print(f"File Content: {file.read()}")
        #return "Files received"
    #else:
        #return "Please send a POST request"



        # Get the uploaded images from the request
        img1 = imgs[0]
        img2 = imgs[1]
        #print(img1)
        #return jsonify(rn) # just for check the print in the terminal

        # Open the images using Pillow
        # before it need :pip install Pillow
        image1 = Image.open(io.BytesIO(img1))
        image2 = Image.open(io.BytesIO(img2))
        #return jsonify(rn) #---> check point

        # chek the images --> convert to array
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        print(img1_array.shape) #--> (500, 375, 3)
        #return jsonify(rn)
        #return img1_array.tobytes(), img2_array.tobytes()



        ############### Model ###############
        resnet = models.resnet18(pretrained=True)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean_torch = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        std_torch = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        resnet.eval()
        criterion = nn.CrossEntropyLoss()

        ### Data:  imagenet with transform ####
        img_exists = True

        if img_exists:
        # I used this codes to load data
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        else:
            pass

        sample_num = 2

        tensor1=test_transform(image1)
        tensor2=test_transform(image2)
        print(tensor1.shape)
        ### Selected Examples
        input_sp = torch.stack([tensor1, tensor2], dim=0)
        targets = torch.tensor([int(lable1), int(lable2)])


        #print_fig((input_sp * std_torch + mean_torch)[:sample_num])


        ########### Saliency ###############
        input_var = input_sp[:sample_num].clone().detach().requires_grad_(True)
        output = resnet(input_var)
        loss = criterion(output, targets[:sample_num])
        loss.backward()

        unary = torch.sqrt(torch.mean(input_var.grad **2, dim=1))
        unary = unary / unary.view(sample_num, -1).max(1)[0].view(sample_num, 1, 1)
        #print_fig(unary)

        unary16 = F.avg_pool2d(unary, 224//16)
        unary16 = unary16 / unary16.view(sample_num, -1).max(1)[0].view(sample_num, 1, 1)
        #print_fig(unary16)

        """### Puzzle Mix"""

        #### Alpha Sweep###
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        beta = 0.8
        gamma = 1.0
        eta = 0.2

        transport = False

        for alpha in np.linspace(0,1,5):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

            #print_fig(output[0] * std_torch + mean_torch)

        ###Beta Sweep###
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        gamma = 0.
        eta = 0.2

        transport = False

        for beta in np.linspace(0,0.8,4):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

            #print_fig(output[0] * std_torch + mean_torch)

        ### Gamma Sweep####
        indices = [1,0]

        n_labels = 3
        block_num = 16

        alpha = 0.5
        beta = 0.2
        eta = 0.2

        transport = False

        for gamma in np.linspace(0,2,4):
            output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                                block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                                neigh_size=2, mean=mean_torch, std=std_torch,
                                transport=transport, t_eps=0.8, t_size=16,
                                device='cpu')

            #print_fig(output[0] * std_torch + mean_torch)

        ###Transport ###
        indices = [1,0]

        n_labels = 2
        block_num = 4

        alpha = 0.4
        beta = 0.2
        gamma = 1.0
        eta = 0.2

        transport = True
        t_eps=0.2
        t_size=224//block_num

        output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                            block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                            neigh_size=2, mean=mean_torch, std=std_torch,
                            transport=transport, t_eps=t_eps, t_size=t_size,
                            device='cpu')

        print_fig(output[0] * std_torch + mean_torch)
        print(output[1])


        return jsonify(rn)# just for return in the POSTMAN
        #'''



@app.route('/')
def hello_world():
    return "Hello World!"


if __name__=="__main__":
    app.run(debug=True)