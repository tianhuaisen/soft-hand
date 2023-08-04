from torchvision.transforms.functional import *

def initiate(label=None):
    if label == "zero":
        shape = torch.zeros(10).unsqueeze(0)
        pose = torch.zeros(48).unsqueeze(0)
    elif label == "uniform":
        shape = torch.from_numpy(np.random.normal(size=[1, 10])).float()
        pose = torch.from_numpy(np.random.normal(size=[1, 48])).float()
    elif label == "01":
        shape = torch.rand(1, 10)
        pose = torch.rand(1, 48)
    else:
        raise ValueError("{} not in ['zero'|'uniform'|'01']".format(label))
    return pose, shape
