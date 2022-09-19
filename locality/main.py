from networks import BiSeNet as model
import torch
import numpy as np
import glob
import argparse
import ipdb
from torchvision import transforms, utils
from PIL import Image
import os
from tqdm import tqdm
import math

remap_list_celebahq = torch.tensor([0, 1, 6, 7, 4, 5, 2, 2, 10, 11, 12, 8, 9, 15, 3, 17, 16, 18, 13, 14]).float()
remap_list = torch.tensor([0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13, 14, 15, 16]).float()

att_channel_dict = {'background': [0],
                    'skin': [1],
                    'nose': [2],
                    'eyes': [4, 5],
                    'eyebrows': [6,7],
                    'mouth': [10, 11, 12],
                    'lips': [11, 12],
                    'hair': [13]}


def id_remap(seg, type='sof'):
    if type == 'sof':
        return remap_list[seg.long()].to(seg.device)
    elif type == 'celebahq':
        return remap_list_celebahq[seg.long()].to(seg.device)

def parsing_img(bisNet, image, to_tensor, argmax=True):
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0).cuda()
        segmap = bisNet(img)[0]
        if argmax:
            segmap = segmap.argmax(1, keepdim=True)
        segmap = id_remap(segmap, 'celebahq')
    return img.cpu().numpy(), torch.squeeze(segmap)

def initFaceParsing(n_classes=20, path=None):
    net = model.BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(path+'/segNet-20Class.pth'))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])
    return net, to_tensor

def face_parsing(img_path, bisNet, to_tensor):
    # img = Image.open(os.path.join(save_dir, 'images512x512', img_path)).convert('RGB')
    tf_resize = transforms.Resize(512)
    img = Image.open(img_path).convert('RGB')
    _, seg_label = parsing_img(bisNet, img.resize((512, 512)), to_tensor, argmax=True)
    seg_mask = seg_label.detach().cpu().numpy()
    return tf_resize(to_tensor(img)), seg_mask



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--original_image_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--edited_image_dir", type=str, default=None,
                        help="The directory to save the segmentation map")
    parser.add_argument("--locality_text", type=str, default=None,
                        help="locality text to measure In-MSE and Out-MSE")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="file to log evaluation output")
     
    args = parser.parse_args()
    
    bisNet, to_tensor = initFaceParsing(path='checkpoints')

    orig_imgs = glob.glob(args.original_image_dir  + "/*.png")
    edited_imgs = glob.glob(args.edited_image_dir  + "/*.png")
    in_mses = []
    out_mses = []
    in_mses_nonan=[]
    out_mses_nonan=[]
    for orig, edit in tqdm(zip(orig_imgs, edited_imgs)):
        orig_image, orig_att = face_parsing(orig, bisNet, to_tensor)
        edit_image, edit_att = face_parsing(edit, bisNet, to_tensor)
        cs = att_channel_dict[args.locality_text]
        orig_masks = []
        edit_masks = []
        for c in cs:
            orig_masks.append(orig_att == c)
            edit_masks.append(edit_att == c)
        orig_mask = np.expand_dims((sum(orig_masks) > 0), axis=0)
        edit_mask = np.expand_dims((sum(edit_masks) > 0), axis=0)
        
        ref_mask = orig_mask if np.sum(orig_mask) >= np.sum(edit_mask) else edit_mask

        ref_mask_area = np.sum(ref_mask)
        total_area = np.prod(ref_mask.shape)

        ## In - MSE
        in_mse = torch.nn.functional.mse_loss(orig_image*ref_mask, edit_image*ref_mask) * total_area / ref_mask_area
        in_mses.append(in_mse.item())
        if not torch.isnan(in_mse):
            in_mses_nonan.append(in_mse.item())
        ## Out - MSE
        out_mse = torch.nn.functional.mse_loss(orig_image * (ref_mask == False), edit_image * (ref_mask == False)) * total_area / (total_area - ref_mask_area)
        out_mses.append(out_mse.item())
        if not torch.isnan(out_mse):
            out_mses_nonan.append(out_mse.item())

    #with open(args.log_dir + "/{}_in_out_mse.txt".format(args.locality_text), 'w') as f:
    with open(args.log_dir, 'w') as f:
        f.write("Evaluation text: {}".format(args.locality_text))
        f.write("\n")
        f.write("Mean in-mse: {:.3f}".format(sum(in_mses_nonan)/len(in_mses_nonan)))
        f.write("\n")
        f.write("Mean out-mse: {:.3f}".format(sum(out_mses_nonan)/len(out_mses_nonan)))
        for i in range(3):
            f.write("\n")
        for i in range(len(orig_imgs)):
            f.write("In-mse: {:.3f} for | {}".format(in_mses[i], orig_imgs[i]))
            f.write("\n")
            f.write("Out-mse: {:.3f} for | {}".format(out_mses[i], orig_imgs[i]))
            f.write("\n")

