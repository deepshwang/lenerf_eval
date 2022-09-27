import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models import resnet
import ipdb

def main():
    # Epoch
    epochs = 50
    thres = 0.9
    # Model configuration
    model = resnet.CelebAMultiClassifier(num_classes=40)
    model.to('cuda')
    #### Dataset configuration
    T = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
    train_dataset = torchvision.datasets.CelebA(root='data', split='train', target_type='attr', transform=T, download=False)
    val_dataset = torchvision.datasets.CelebA(root='data', split='valid', target_type='attr', transform=T, download=False)
    test_dataset = torchvision.datasets.CelebA(root='data', split='test', target_type='attr', transform=T, download=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)


    model.load_state_dict(torch.load("checkpoints/best_net.pth"))
    corrects = []
    for i, (imgs, labels) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        imgs = imgs.to('cuda')
        labels = labels.to('cuda')
        logits = model(imgs)
        correct = labels == (logits > thres)
        corrects.append(correct)
    corrects = torch.cat(tuple(corrects), dim=0)
    corrects = torch.sum(corrects, dim=0)
    correct_percentage = corrects / len(test_dataset)
    correct_percentage_average = torch.mean(correct_percentage)
    with open("classification_training_test_stats.txt", 'w') as f:
        f.write("Mean correct classification percentage: {:.2f}".format(float(correct_percentage_average.cpu().numpy())* 100))
        for i in range(correct_percentage.shape[0]):    
            f.write("\n")
            f.write("class {}: {:.2f}".format(i, float(correct_percentage[i].cpu().numpy())* 100))
        

if __name__ == "__main__":
    main()