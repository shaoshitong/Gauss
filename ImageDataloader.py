from torch.utils.data import Dataset
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_trainsform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
def default_loader(path):
    return Image.open(path).convert('RGB')
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
def default_reader(file_path):
    imlist = []
    print(file_path)
    with open(file_path, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist
def save_file_txt(flist,file_name):
    with open(file_name,'w') as rf:
        for f in flist:
            rf.writelines(f+'\n')
class ImageFilelist(Dataset):
    def __init__(self, file, transform=train_transform,
                 flist_reader=default_reader, loader=default_loader,save_file=save_file_txt):
        self.root = ""
        self.flist=make_dataset("./train")
        self.save_file=save_file(self.flist,file)
        self.imlist = flist_reader(file)
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([
            str((path.split("\\")[-1]).split('_')[0])+str((path.split("\\")[-1]).split('_')[1])
            for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[str((impath.split("\\")[-1]).split('_')[0])+str((impath.split("\\")[-1]).split('_')[1])]) for impath in self.imlist]
    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)