import torch
import utility
import scipy.misc as misc
from option import args
from model.arbrcan import ArbRCAN
import imageio
import numpy as np


if __name__ == '__main__':
    if args.n_GPUs > 0:
        device = 'cuda:0'
    else:
        device = 'cpu'

    model = ArbRCAN(args).to(device)
    ckpt = torch.load('experiment/ArbRCAN/model/model_'+str(args.resume)+'.pt', map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # load lr image
    lr = imageio.imread(args.dir_img)
    lr = np.array(lr)
    lr = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(device)

    # model is trained on scale factors in range [1, 4]
    # one can also try out-of-distribution scale factors but the results may be not very promising
    assert args.sr_size[0] / lr.size(2) > 1 and args.sr_size[0] / lr.size(2) <= 4
    assert args.sr_size[1] / lr.size(3) > 1 and args.sr_size[1] / lr.size(3) <= 4

    with torch.no_grad():
        scale = args.sr_size[0] / lr.size(2)
        scale2 = args.sr_size[1] / lr.size(3)
        model.set_scale(scale, scale2)
        sr = model(lr)

        sr = utility.quantize(sr, args.rgb_range)
        sr = sr.data.mul(255 / args.rgb_range)
        sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
        filename = 'experiment/quick_test/results/{}x{}'.format(int(args.sr_size[0]), int(args.sr_size[1]))
        misc.imsave('{}.png'.format(filename), sr)
