from options.train_options import TrainOptions
from util.visualizer import Visualizer
from torch.optim import lr_scheduler
from data import create_dataset
from models import create_model
import util.util as util
import time
import os


def save_result(path, result):
    tmp = open(path, mode='w')
    tmp.write(result)
    tmp.close()


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter_stage2) / float(opt.niter_decay_stage2 + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.stage = 2
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = opt.niter_decay_stage2 + opt.niter_stage2
    iter_step = 10

    result_path = os.path.join('./results', opt.name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for i, data in enumerate(dataset):  # inner loop within one epoch
        epoch_start_time = time.time()
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        scheduler = get_scheduler(model.optimizer, opt)
        visualizer = Visualizer(opt)
        visualizer.reset()

        now_iter = 0
        tmp_t0 = 0
        threshold = 1e-4
        while now_iter < total_iters:
            iter_start_time = time.time()
            for j in range(iter_step):
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), i, False)

            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(i, now_iter + 1, losses, t_comp, 0)
            if opt.display_id > 0:
                visualizer.plot_current_losses(i, float(now_iter + 1) / total_iters, losses)

            tmp_t1 = 1 - model.loss_G_MEFSSIM
            if abs(tmp_t1 - tmp_t0) < threshold:
                now_iter += 1
                scheduler.step()  # update learning rates at the end of every epoch.
                lr = model.optimizer.param_groups[0]['lr']
                print('learning rate = %.7f' % lr)
            tmp_t0 = tmp_t1

        print('saving the results at the end of image %d, iters %d' % (i, total_iters))
        
        tmp = model.get_current_visuals()['fake_B']
        fake = util.tensor2im(tmp)
        save_path_prefix = os.path.join(result_path, data['image_name'][0])
        util.save_image(fake, save_path_prefix+'.png', aspect_ratio=1.0)
        print(model.get_current_latent()[0])
        save_result(save_path_prefix+'.txt', str(model.get_current_latent()[0]))
        epoch_time = time.time() - epoch_start_time
        save_result(save_path_prefix + '_time.txt', str(epoch_time))
        print('End of image %d / %d \t Time Taken: %d sec' % (i, dataset_size, epoch_time))
        r = 1 - model.loss_G_MEFSSIM
        save_result(save_path_prefix + '_ssim.txt', str(r))
