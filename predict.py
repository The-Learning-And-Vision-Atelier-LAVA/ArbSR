import torch
import utility
import argparse
from model.arbrcan import ArbRCAN
import imageio
import tempfile
from pathlib import Path
import numpy as np
import PIL.Image
import cog


class Predictor(cog.Predictor):
    def setup(self):
        self.device = torch.device('cuda:0')
        self.args = get_args()
        self.args.n_GPUs = 1
        self.args.resume = 150
        self.model = ArbRCAN(self.args).to(self.device)
        ckpt = torch.load('experiment/ArbRCAN/model/model_' + str(self.args.resume) + '.pt', map_location=self.device)
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    @cog.input(
        "target_width",
        type=int,
        help="width of the target image, set 1-4 times of the input width",
    )
    @cog.input(
        "target_height",
        type=int,
        help="height of the target image, set 1-4 times of the input height",
    )
    def predict(self, image, target_width, target_height):
        # load lr image

        input_path = str(image)
        lr = imageio.imread(input_path)
        # nn.AvgPool2d(2) in SA_adapt class from model/arbrcan.py will cause error when input width or height is an odd
        # number, so crop to the closest even number first
        if lr.shape[0] % 2 == 1 or lr.shape[1] % 2 == 1:
            ori_image = PIL.Image.open(input_path)
            processed_image = ori_image.crop((0, 0, 2 * (lr.shape[1] // 2), 2 * (lr.shape[0] // 2)))
            # for replicate, webcam input might be rgba, convert to rgb first
            if lr.shape[-1] == 4:
                processed_image = processed_image.convert('RGB')
            input_path = 'even_rgb.png'
            imageio.imwrite(input_path, processed_image)
            lr = imageio.imread(input_path)
        else:
            if lr.shape[-1] == 4:
                rgba_image = PIL.Image.open(input_path)
                processed_image = rgba_image.convert('RGB')
                input_path = 'even_rgb.png'
                imageio.imwrite(input_path, processed_image)
                lr = imageio.imread(input_path)

        lr = np.array(lr)
        lr = torch.Tensor(lr).permute(2, 0, 1).contiguous().unsqueeze(0).to(self.device)

        # model is trained on scale factors in range [1, 4]
        # one can also try out-of-distribution scale factors but the results may be not very promising
        sr_size = f'{target_height}+{target_width}'
        self.args.sr_size = list(map(lambda x: float(x), sr_size.split('+')))
        scale = self.args.sr_size[0] / lr.size(2)  # height
        scale2 = self.args.sr_size[1] / lr.size(3)  # width

        assert 1 < scale <= 4, f'input width is {lr.size(2)}, output width is suggested to set to be 1-4 times of ' \
                               f'the input width'
        assert 1 < scale2 <= 4, f'input height is {lr.size(3)}, output height is suggested to set to be 1-4 times ' \
                                f'of the input height'
        with torch.no_grad():
            self.model.set_scale(scale, scale2)
            sr = self.model(lr)

            sr = utility.quantize(sr, self.args.rgb_range)
            sr = sr.data.mul(255 / self.args.rgb_range)
            sr = sr[0, ...].permute(1, 2, 0).cpu().numpy()
            out_path = Path(tempfile.mkdtemp()) / "out.png"
            imageio.imsave(str(out_path), sr)
            imageio.imsave('ooo.png', sr)
        return out_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enables debug mode')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=2,
                        help='number of threads for data loading')
    parser.add_argument('--cpu', type=bool, default=False,
                        help='use cpu only')
    parser.add_argument('--n_GPUs', type=int, default=2,
                        help='number of GPUs')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')

    # Data specifications
    parser.add_argument('--dir_data', type=str, default='F:/LongguangWang/Data',
                        help='dataset directory')
    parser.add_argument('--data_train', type=str, default='DIV2K',
                        help='train dataset name')
    parser.add_argument('--data_test', type=str, default='Set5',
                        help='test dataset name')
    parser.add_argument('--data_range', type=str, default='1-800/801-810',
                        help='train/test data range')
    parser.add_argument('--ext', type=str, default='sep',
                        help='dataset file extension')
    parser.add_argument('--asymm', type=bool, default=True,
                        help='use asymmetric scale factors (only used during training phase)')
    parser.add_argument('--scale', type=str, default='',
                        help='super resolution scale')
    parser.add_argument('--scale2', type=str, default='',
                        help='super resolution scale2')
    parser.add_argument('--patch_size', type=int, default=50,
                        help='input patch size')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--chop', default=False,
                        help='enable memory-efficient forward')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')

    # Model specifications
    parser.add_argument('--model', default='ArbRCAN',
                        help='model name')
    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--pre_train', type=str, default='model/RCAN_BIX4.pt',
                        help='pre-trained model directory')
    parser.add_argument('--extend', type=str, default='.',
                        help='pre-trained model directory')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    parser.add_argument('--dilation', action='store_true',
                        help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'),
                        help='FP precision for test (single | half)')

    # Training specifications
    parser.add_argument('--reset', action='store_true',
                        help='reset the training')
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training')
    parser.add_argument('--split_batch', type=int, default=1,
                        help='split the batch into smaller chunks')
    parser.add_argument('--self_ensemble', action='store_true',
                        help='use self-ensemble method for test')
    parser.add_argument('--test_only', type=bool, default=False,
                        help='set this option to test the model')
    parser.add_argument('--gan_k', type=int, default=1,
                        help='k value for adversarial loss')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=int, default=20,
                        help='learning rate decay per N epochs')
    parser.add_argument('--decay_type', type=str, default='step',
                        help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='ADAM beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='ADAM beta2')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='resume from the snapshot, and the start_epoch')

    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1',
                        help='loss function configuration')
    parser.add_argument('--skip_threshold', type=float, default='1e6',
                        help='skipping batch that has large error')

    # Log specifications
    parser.add_argument('--save', type=str, default='ArbRCAN',
                        help='file name to save')
    parser.add_argument('--load', type=str, default='.',
                        help='file name to load')
    parser.add_argument('--resume', type=int, default=0,
                        help='resume from specific checkpoint')
    parser.add_argument('--save_models', action='store_true',
                        help='save all intermediate models')
    parser.add_argument('--print_every', type=int, default=200,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_results', default=False,
                        help='save output results')

    # Quick test specifications
    parser.add_argument('--dir_img', type=str, default='experiment/quick_test/img_004.png',
                        help='image directory for quick test')
    parser.add_argument('--sr_size', default='512+512',
                        help='size of SR images for quick test')

    # ('') is required for cog so args parse needs to be re-defined in predict.py
    args = parser.parse_args('')

    if args.scale == '' or args.scale2 == '':
        # asymmetric mode: non-integer scale factors + asymmetric scale factors
        if args.asymm:
            args.scale = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                1.5, 1.5, 1.5, 1.5, 1.5,
                2.0, 2.0, 2.0, 2.0, 2.0,
                2.5, 2.5, 2.5, 2.5, 2.5,
                3.0, 3.0, 3.0, 3.0, 3.0,
                3.5, 3.5, 3.5, 3.5, 3.5,
                4.0, 4.0, 4.0, 4.0, 4.0,
            ]
            args.scale2 = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
                2.0, 2.5, 3.0, 3.5, 4.0,
                1.5, 2.5, 3.0, 3.5, 4.0,
                1.5, 2.0, 3.0, 3.5, 4.0,
                1.5, 2.0, 2.5, 3.5, 4.0,
                1.5, 2.0, 2.5, 3.0, 4.0,
                1.5, 2.0, 2.5, 3.0, 3.5,
            ]
        # symmetric mode: only non-integer scale factors
        else:
            args.scale = [
                1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
                2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0
            ]
            args.scale2 = args.scale
    else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
        args.scale2 = list(map(lambda x: float(x), args.scale2.split('+')))

    assert len(args.scale) == len(args.scale2)

    if args.epochs == 0:
        args.epochs = 1e8

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    return args
