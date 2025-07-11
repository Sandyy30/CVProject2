import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP
from model import U2NETE

import time
import matplotlib.pyplot as plt
import csv
import argparse

# --------- argparse for command-line argument ---------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="u2net", choices=["u2net", "u2netp", "u2nete"], help="Model type: u2net | u2netp | u2nete")
args = parser.parse_args()
model_name = args.model

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss


# ------- 2. set the directory of training dataset --------

# model_name = 'u2net' # ‘u2net' 'u2netp'
print("Model: ", model_name)

tra_data_dir = os.path.join(os.getcwd(), 'train1000' + os.sep)
tra_image_dir = os.path.join('Images' + os.sep)
tra_label_dir = os.path.join('Masks' + os.sep)
val_data_dir = os.path.join(os.getcwd(), 'val200' + os.sep)
val_image_dir = os.path.join('Images' + os.sep)
val_label_dir = os.path.join('Masks' + os.sep)

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
os.makedirs(model_dir, exist_ok=True)

epoch_num = 50
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(tra_data_dir + tra_image_dir + '*' + image_ext)
val_img_name_list = glob.glob(val_data_dir + val_image_dir + '*' + image_ext)

tra_lbl_name_list = []
val_lbl_name_list = []

for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(tra_data_dir + tra_label_dir + imidx + label_ext)

for img_path in val_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	val_lbl_name_list.append(val_data_dir + val_label_dir + imidx + label_ext)


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("validation images: ", len(val_img_name_list))
print("validation labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
val_num = len(val_img_name_list)

train_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

val_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
val_dataloader = DataLoader(val_dataset, batch_size=batch_size_train, shuffle=False, num_workers=1)


# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3,1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)
elif(model_name=="u2nete"):
    net = U2NETE(3,1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 20 # save the model every 20 iterations

train_epoch_losses = []
train_epoch_target_losses = []
val_epoch_losses = []
val_epoch_target_losses = []

for epoch in range(0, epoch_num):
    start_time = time.time()
    net.train()

    for i, data in enumerate(train_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    avg_train_loss = running_loss / ite_num4val
    avg_train_tar_loss = running_tar_loss / ite_num4val
	
    net.eval()
    val_running_loss = 0.0
    val_running_tar_loss = 0.0
	
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data['image'], data['label']
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
	
            if torch.cuda.is_available():
                inputs_v, labels_v = inputs.cuda(), labels.cuda()
            else:
                inputs_v, labels_v = inputs, labels
    
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            val_loss2, val_loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
	
            val_running_loss += val_loss.data.item()
            val_running_tar_loss += val_loss2.data.item()

    end_time = time.time()  # ⏱️ End timing
    elapsed_time = end_time - start_time

    avg_val_loss = val_running_loss / len(val_dataloader)
    avg_val_tar_loss = val_running_tar_loss / len(val_dataloader)
    print(f"[Epoch {epoch + 1}/{epoch_num}] Train Loss: {avg_train_loss:.6f}, Train Target Loss: {avg_train_tar_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Validation Target Loss: {avg_val_tar_loss:.6f}, Time Taken: {elapsed_time:.2f}s")

    train_epoch_losses.append(avg_train_loss)
    train_epoch_target_losses.append(avg_train_tar_loss)
    val_epoch_losses.append(avg_val_loss)
    val_epoch_target_losses.append(avg_val_tar_loss)

    # Reset epoch accumulators
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0

# Save final model after all epochs
final_model_path = os.path.join(model_dir, model_name + "_final.pth")
torch.save(net.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

with open(os.path.join(model_dir, 'loss_log.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'Train Loss', 'Train Target Loss', 'Val Loss', 'Val Target Loss'])
    for i in range(len(train_epoch_losses)):
        writer.writerow([
            i + 1,
            train_epoch_losses[i],
            train_epoch_target_losses[i],
            val_epoch_losses[i],
            val_epoch_target_losses[i]
        ])

print(f"Losses saved to {os.path.join(model_dir, 'loss_log.csv')}")
