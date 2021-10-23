import os
import torch
import utility
import scipy.misc as misc
from option import args
from model.arbrcan import ArbRCAN
import imageio
import tempfile
from pathlib import Path
import numpy as np

import cog


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda:0')
        args.n_GPUs = 1
        args.sr_size = '512+512'
        args.resume = 150
        self.model = ArbRCAN(args).to(self.device)
        ckpt = torch.load('experiment/ArbRCAN/model/model_' + str(args.resume) + '.pt', map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    def predict(self, image):
        # load lr image
        lr = imageio.imread(str(image))
        lr = np.array(lr)
        lr = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)

        # model is trained on scale factors in range [1, 4]
        # one can also try out-of-distribution scale factors but the results may be not very promising
        scale = int(args.sr_size.split('+')[0]) / lr.size(2)
        scale2 = int(args.sr_size.split('+')[1]) / lr.size(3)
        assert 1 < scale <= 4
        assert 1 < scale2 <= 4

        with torch.no_grad():
            self.model.set_scale(scale, scale2)
            sr = self.model(lr)

            sr = utility.quantize(sr, args.rgb_range)
            sr = sr.data.mul(255 / args.rgb_range)
            sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            imageio.imsave(str(out_path), sr)

        return out_path
