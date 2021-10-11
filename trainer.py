import os
import math
import matplotlib
matplotlib.use('TKAgg')
import utility
import torch
import numpy as np
from decimal import Decimal


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # train on integer scale factors (x2, x3, x4) for 1 epoch to maintain stability
        if epoch == 1:
            self.loader_train.dataset.first_epoch = True
            # adjust learning rate
            lr = 5e-5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        # train on all scale factors for remaining epochs
        else:
            self.loader_train.dataset.first_epoch = False
            # adjust learning rate
            lr = self.args.lr * (2 ** -(epoch // 30))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            scale = hr.size(2) / lr.size(2)
            scale2 = hr.size(3) / lr.size(3)
            timer_data.hold()
            self.optimizer.zero_grad()

            # inference
            self.model.get_model().set_scale(scale, scale2)
            sr = self.model(lr)

            # loss function
            loss = self.loss(sr, hr)

            # backward
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        target = self.model
        torch.save(
            target.state_dict(),
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )

    def test(self):
        self.model.eval()

        with torch.no_grad():
            for idx_scale, _ in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                scale = self.args.scale[idx_scale]
                scale2 = self.args.scale2[idx_scale]

                eval_psnr = 0
                eval_ssim = 0

                for idx_img, (lr, hr, filename, _) in enumerate(self.loader_test):
                    filename = filename[0]

                    # prepare LR & HR images
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    lr, hr = self.crop_border(lr, hr, scale, scale2)

                    # inference
                    self.model.get_model().set_scale(scale, scale2)
                    sr = self.model(lr)

                    # evaluation
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, [scale, scale2], self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        eval_ssim += utility.calc_ssim(
                            sr, hr, [scale, scale2],
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    # save SR results
                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                if scale == scale2:
                    print('[{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
                else:
                    print('[{} x{}/x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.data_test,
                        scale,
                        scale2,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))


    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def crop_border(self, img_lr, img_hr, scale, scale2):
        N, C, H_lr, W_lr = img_lr.size()
        N, C, H_hr, W_hr = img_hr.size()
        H = H_lr if round(H_lr * scale) <= H_hr else math.floor(H_hr / scale)
        W = W_lr if round(W_lr * scale2) <= W_hr else math.floor(W_hr / scale2)

        step = []
        for s in [scale, scale2]:
            if s == int(s):
                step.append(1)
            elif s * 2 == int(s * 2):
                step.append(2)
            elif s * 5 == int(s * 5):
                step.append(5)
            elif s * 10 == int(s * 10):
                step.append(10)
            elif s * 20 == int(s * 20):
                step.append(20)
            elif s * 50 == int(s * 50):
                step.append(50)

        H_new = H // step[0] * step[0]
        if H_new % 2 == 1:
            H_new = H // (step[0] * 2) * step[0] * 2

        W_new = W // step[1] * step[1]
        if W_new % 2 == 1:
            W_new = W // (step[1] * 2) * step[1] * 2

        img_lr = img_lr[:, :, :H_new, :W_new]
        img_hr = img_hr[:, :, :round(scale * H_new), :round(scale2 * W_new)]

        return img_lr, img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
