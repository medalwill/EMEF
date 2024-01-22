from util.visualizer import save_images_evaluate
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
import torch
import os


def save_result(path, result):
    tmp = open(path, mode='w')
    tmp.write(result)
    tmp.close()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.stage = 1
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset) 
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    
    model.eval()
    total_ssim = 0
    total_psnr = 0
    total_l1 = 0
    L1LOSS = torch.nn.L1Loss()
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        print('processing (%04d)-th image... %s' % (i, img_path))
        ssim, psnr, l1 = save_images_evaluate(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio,
                                              width=opt.display_winsize, L1LOSS=L1LOSS, modelname=opt.model)
        total_ssim += ssim
        total_psnr += psnr
        total_l1 += l1
    average_ssim = total_ssim / dataset_size
    average_psnr = total_psnr / dataset_size
    average_l1 = total_l1 / dataset_size
    
    result_dir = os.path.join(web_dir, 'result.txt')
    result_str = 'ssim = %f, psnr = %f, l1 = %f' % (average_ssim, average_psnr, average_l1)
    print(result_str)
    print(f'Save result to {result_dir}')
    save_result(result_dir, result_str)
    webpage.save()  # save the HTML
