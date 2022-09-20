import ipdb
import pickle
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
from models import resnet
import glob
from PIL import Image
from tqdm import tqdm
import os

attr_dict = {'5_o_clock_shadow': 0,
             'arched_eyebrows': 1,
             'attractive': 2,
             'bags_under_eyes': 3,
             'bald': 4,
             'bangs': 5,
             'big_lips': 6,
             'big_nose': 7, 
             'black_hair': 8, 
             'blond_hair': 9,
             'blurry': 10,
             'brown_hair': 11,
             'bushy_eyebrows': 12,
             'chubby': 13,
             'double_chin': 14,
             'eyeglasses': 15,
             'goatee': 16,
             'gray_hair': 17,
             'heavy_makeup': 18, 
             'high_cheekbones': 19,
             'male': 20,
             'mouth_slightly_open': 21, 
             'mustache': 22, 
             'narrow_eyes': 23, 
             'no_beard': 24,
             'oval_face': 25,
             'pale_skin': 26, 
             'pointy_nose': 27,
             'receding_hairline': 28, 
             'rosy_cheeks' : 29,
             'sideburns': 30,
             'smiling': 31,
             'straight_hair': 32, 
             'curly_hair': 33, #Wavy_Hair
             'wearing_earrings': 34, 
             'wearing_hat': 35,
             'wearing_lipstick': 36, 
             'wearing_necklace': 37,
             'wearing_necktie': 38,
             'young': 39}

class AttributePairDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dir, edit_dir):
        self.orig_imgs = glob.glob(orig_dir + "/*.png")
        self.edited_imgs = glob.glob(edit_dir + "/*.png")
        self.T = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

    def __getitem__(self, idx):
        orig_img = Image.open(self.orig_imgs[idx])
        edit_img = Image.open(self.edited_imgs[idx])
        return self.T(orig_img), self.T(edit_img)

    def __len__(self):
        return len(self.orig_imgs)

def main(args):
    # model 
    model = resnet.CelebAMultiClassifier(num_classes=40)
    model.load_state_dict(torch.load("checkpoints/best_net.pth"))
    model.to('cuda')
    model.eval()
    
    # data
    dataset = AttributePairDataset(args.original_image_dir, args.edited_image_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    eval_thres = float(args.eval_thres)
    orig_probs = []
    edit_probs = []
    for i, (orig, edit) in tqdm(enumerate(dataloader), total=len(dataloader)):
        orig = orig.to('cuda')
        edit = edit.to('cuda')
        attr_idx = attr_dict[args.attribute_text]
        orig_logit = model(orig)
        edit_logit = model(edit)
        orig_prob = orig_logit[:, attr_idx]
        edit_prob = edit_logit[:, attr_idx]
        orig_probs.append(orig_prob)
        edit_probs.append(edit_prob)
    orig_probs = torch.cat(tuple(orig_probs))
    edit_probs = torch.cat(tuple(edit_probs))
    orig_neg_idx = torch.where(orig_probs < (1-eval_thres))[0]
    
    # Redefine a subset where the original is definitely negative given an attribute
    orig_probs_valid = orig_probs[orig_neg_idx]
    edit_probs_valid = edit_probs[orig_neg_idx]
    edit_pos_idx = torch.where(edit_probs_valid > eval_thres)[0]
    success_rate = edit_pos_idx.shape[0] / orig_probs_valid.shape[0]
    
    #with open(os.path.join(args.log_dir, "eval_stats_{}.txt".format(args.attribute_text)), 'w') as f:
    with open(args.log_dir, 'w') as f:
        f.write("Average edit success rate: {:.3f}".format(success_rate))
        f.write("\n")
        for i in range(len(dataset)):
            if i not in orig_neg_idx:
                msg = 'Original image is not completely negative'
            else:
                if edit_probs[i] > eval_thres:
                    msg = 'Success!\n'
                else:
                    msg = 'Failure\n'
                msg += 'logit before edit: {:.2f}\n'.format(orig_probs[i])
                msg += 'logit after edit: {:.2f}'.format(edit_probs[i])
            f.write("{} : ".format(dataset.orig_imgs[i].split("/")[-1]))
            f.write("\n")
            f.write(msg)
            for i in range(2):
                f.write("\n")

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--original_image_dir", type=str, default=None,
                        help="The directory of the images to be inverted")
    parser.add_argument("--edited_image_dir", type=str, default=None,
                        help="The directory to save the segmentation map")
    parser.add_argument("--attribute_text", type=str, default=None,
                        help="locality text to measure In-MSE and Out-MSE")
    parser.add_argument("--eval_thres", type=str, default=None,
                        help="evaluation threshold")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="file to log evaluation output")
    args = parser.parse_args()
    with torch.no_grad():
        main(args)