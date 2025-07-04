import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
from model import U2NETE


class Sobel(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[1.0, 0.0, -1.0], 
                           [2.0, 0.0, -2.0], 
                           [1.0, 0.0, -1.0]])
        Gy = torch.tensor([[1.0, 2.0, 1.0], 
                           [0.0, 0.0, 0.0], 
                           [-1.0, -2.0, -1.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        # Convert RGB to grayscale
        if img.shape[1] == 3:
            gray = 0.2989 * img[:, 0:1, :, :] + \
                   0.5870 * img[:, 1:2, :, :] + \
                   0.1140 * img[:, 2:3, :, :]
        else:
            gray = img

        x = self.filter(gray)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x + 1e-6)  # avoid sqrt(0)

        if self.normalize:
            x_min = x.amin(dim=(2,3), keepdim=True)
            x_max = x.amax(dim=(2,3), keepdim=True)
            x = (x - x_min) / (x_max - x_min + 1e-6)

        return x

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    save_path = os.path.join(d_dir, imidx + '.png')
    imo.save(save_path)

def main():
    model_name = 'u2nete'  # u2net / u2netp / u2nete

    image_dir = os.path.join(os.getcwd(), 'test500')
    prediction_dir = os.path.join(os.getcwd(), 'test_result')
    model_path = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '_final.pth')

    img_name_list = glob.glob(os.path.join(image_dir, 'Images', '*.*'))
    print('Test images:', len(img_name_list))

    # --------- 1. dataloader ---------
    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)
        ])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 2. model define ---------
    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        net = U2NETP(3, 1)
    elif model_name == 'u2nete':
        net = U2NETE(3, 1)  # still pass 3 here, model splits 3+1 inside

    # Load model
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    # --------- 3. define Sobel layer ---------
    sobel_layer = Sobel(normalize=True)
    if torch.cuda.is_available():
        sobel_layer = sobel_layer.cuda()

    # --------- 4. inference ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image'].type(torch.FloatTensor)
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        # ✅ Generate Sobel edge
        with torch.no_grad():
            edge = sobel_layer(inputs_test)  # [1, 1, H, W]
            input_4ch = torch.cat([inputs_test, edge], dim=1)  # [1, 4, H, W]

        # ✅ Forward pass
        with torch.no_grad():
            d0, d1, d2, d3, d4, d5, d6 = net(input_4ch)

        # Normalize prediction
        pred = d0[:, 0, :, :]
        pred = normPRED(pred)

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d0, d1, d2, d3, d4, d5, d6

    print("Saved predictions:")
    for file in os.listdir(prediction_dir):
        print(file)


if __name__ == "__main__":
    main()
