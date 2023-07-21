import torch
import time
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor
from dataset import MEFdataset
from option import args
from sci import MEF_Network
from utils import saveImage


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = Compose([ToTensor(), ])
    test_set = MEFdataset(args.dir_test, transform=transforms)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)
    step = 0
    times = 0

    # load model
    model_f = MEF_Network().to(device)
    model_f.load_state_dict(torch.load(args.model_pth))
    model_f.eval()

    # test
    for over, under in test_loader:
        over = over.to(device)
        under = under.to(device)
        with torch.no_grad():
            start = time.time()
            fused_output = model_f(over, under)
            end = time.time()
            t = end - start
            times = times + t
            save_path = test_set.save_imgs(step)
            saveImage(fused_output, save_path)


        step = step + 1
    print('total runtime：{}; '.format(times), 'average runtime: {}'.format(times / (step+1)))



if __name__ == '__main__':
    main()
