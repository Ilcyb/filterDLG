from os.path import expandvars
import os
import math
import time
import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import torch
from torchvision import datasets 

sys.path.append('..')

from utils import *  
from models import * 

torch.manual_seed(50)

def get_real_datas(net, save_dir, config):
    dataset = config['dataset']
    participants = config['participants']
    batch_size = config['batch_size']

    dst = None
    if dataset == 'cifar10':
        dst = datasets.CIFAR10(base_config['dataset']['cifar10_path'], download=True)
    elif dataset == 'mnist':
        dst = datasets.MNIST(base_config['dataset']['mnist_path'], download=True)
    elif dataset == 'cifar100':
        dst = datasets.CIFAR100(base_config['dataset']['cifar100_path'], download=True)
    elif dataset == 'svhn':
        dst = datasets.SVHN(base_config['dataset']['svhn_path'], download=True)
    elif dataset == 'lfw':
        dst = datasets.ImageFolder(base_config['dataset']['lfw_path'])

    if config['truth_imgs'] is None:
        img_idxs = [np.random.choice(range(10000), batch_size) for _ in range(participants)]
    else:
        img_idxs = [config['truth_imgs'][:batch_size] for _ in range(participants)]
    leak_data_size = tp(dst[0][0]).size()

    data_shape = (participants, batch_size, *leak_data_size)
    label_shape = (participants, batch_size)
    gt_data = torch.randn(data_shape).to(device)
    gt_label = torch.zeros(label_shape).long().to(device)
    gt_onehot_label = []
    for i in range(participants):
        labels = []
        for j in range(batch_size):
            gt_data[i][j] = tp(dst[img_idxs[i][j]][0]).to(device)
            labels.append(dst[img_idxs[i][j]][1])
        gt_label[i] = torch.Tensor(labels).long().to(device)
        gt_onehot_label.append(label_to_onehot(gt_label[i], num_classes=config['num_classes']))
    label_onehot_shape = (participants, *gt_onehot_label[0].size())

    # compute original gradient 
    total_dy_dx = []
    for i in range(participants):
        out = net(gt_data[i])
        y = criterion(out, gt_onehot_label[i])
        dy_dx = torch.autograd.grad(y, net.parameters())

        # share the gradients with other clients
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        for j in range(len(original_dy_dx)):
            if len(total_dy_dx) <= j:
                total_dy_dx.append(original_dy_dx[j])
            else:
                total_dy_dx[j] += original_dy_dx[j]

    mean_dy_dx = []
    for i in range(len(total_dy_dx)):
        mean_dy_dx.append(total_dy_dx[i] / participants)

    for i in range(participants):
        for j in range(batch_size):
            save_tensor_img(save_dir, 'truth_img{}-{}'.format(i + 1, j + 1), gt_data[i][j].cpu())
        save_tensor_img(save_dir, 'truth_img_grid', gt_data[i], True)

    return gt_data, gt_onehot_label, mean_dy_dx, data_shape, label_onehot_shape


def cpl_patterned(data_shape):
    if data_shape[2] % 2 != 0 or data_shape[3] % 2 != 0:
        raise ValueError('[{}*{}] cannot cpl patterned'.format(data_shape[2], data_shape[3]))
    dummy = torch.zeros(data_shape)
    channel_templates = []
    template_w = (int)(data_shape[2] / 2)
    template_h = (int)(data_shape[3] / 2)
    for c in range(data_shape[1]):
        channel_templates.append(torch.rand(template_w * template_h))
    for bs in range(data_shape[0]):
        for c in range(data_shape[1]):
            count = 0
            for i in range(template_w):
                for j in range(template_h):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w, data_shape[2]):
                for j in range(template_h):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w):
                for j in range(template_h, data_shape[3]):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1

            count = 0
            for i in range(template_w, data_shape[2]):
                for j in range(template_h, data_shape[3]):
                    dummy[bs][c][i][j] = channel_templates[c][count]
                    count += 1
    return dummy.to(device).requires_grad_(True)


def cpl_rgb(data_shape):
    return torch.ones(data_shape).to(device).requires_grad_(True)


def cpl_dark(data_shape):
    return torch.zeros(data_shape).to(device).requires_grad_(True)


def gan_dummy(data_shape, participant, batch_size, config, generate_model):
    dummies = []
    for i in range(batch_size):
        if config['dataset'] == 'mnist':
            # z = Variable(Tensor(np.random.normal(0, 1, (1, 100))))
            z = torch.randn(1, 100, device=device)
        elif config['dataset'] == 'cifar10':
            z = torch.randn(1, 100, 1, 1, device=device)
        elif config['dataset'] == 'svhn':
            z = torch.randn((1, 100), device=device)
        elif config['dataset'] == 'cifar100':
            z = torch.randn(1, 100, 1, 1, device=device)
        elif config['dataset'] == 'lfw':
            z = torch.randn(1, 100, 1, 1, device=device)
        dummy_data = gtp(tt(generate_model(z).squeeze())).view(1, *(data_shape[2:]))
        dummies.append(dummy_data)
    return torch.cat(dummies, 0).to(device).requires_grad_(True)


# generate dummy data and label
def generate_dummy_datas(save_dir, config, data_shape, label_onehot_shape,
                         generate_model, generate_models, gt_onehot_labels=None):
    dummy_datas = []
    dummy_labels = []
    participants = config['participants']
    batch_size = config['batch_size']

    for i in range(participants):
        if config['init_method'] == 'cpl-rgb':
            dummy_datas.append(cpl_rgb(data_shape[1:]))
            dummy_labels.append(cpl_rgb(label_onehot_shape[1:]))
        elif config['init_method'] == 'cpl-dark':
            dummy_datas.append(cpl_rgb(data_shape[1:]))
            dummy_labels.append(cpl_rgb(label_onehot_shape[1:]))
        elif config['init_method'] == 'cpl-patterned':
            dummy_datas.append(cpl_patterned(data_shape[1:]))
            dummy_labels.append(torch.randn(label_onehot_shape[1:]).to(device).requires_grad_(True))
        elif config['init_method'] == 'gan':
            dummy_datas.append(gan_dummy(data_shape, i, batch_size, config, generate_model))
            dummy_labels.append(gt_onehot_labels[i].detach().to(device).requires_grad_(True))
        else:
            dummy_datas.append(torch.rand(data_shape[1:]).to(device).requires_grad_(True))
            dummy_labels.append(torch.randn(label_onehot_shape[1:]).to(device).requires_grad_(True))

    for i in range(participants):
        for j in range(batch_size):
                save_tensor_img(save_dir, 'dummy_img{}-{}'.format(i + 1, j + 1), dummy_datas[i][j].cpu())
        save_tensor_img(save_dir, 'dummy_img_grid', dummy_datas[i], True)

    return dummy_datas, dummy_labels


def recover(save_dir, config, net, gt_data, dummy_datas, dummy_labels, mean_dy_dx):
    dummies = []
    history = []
    loss = []
    psnrs = []
    recover_procedure = []

    participants = config['participants'] or 1
    batch_size = config['batch_size'] or 1
    iters = config['iters'] or 10000
    step_size = config['step_size'] or 1000
    lr = config['lr'] or 0.02
    optim = config['optim'] or 'adam'
    norm_rate = config['norm_rate']
    smooth_direction = config['smooth_direction']
    procedure_save = config['procedure_save']

    for i in range(participants):
        dummies.append(dummy_datas[i])
        dummies.append(dummy_labels[i])
        _ = []
        __ = []
        for j in range(batch_size):
            _.append([])
            __.append([])
        history.append(_)
        recover_procedure.append(__)

    optimizer = None
    if optim == 'adam':
        optimizer = torch.optim.Adam(dummies, lr=lr)
    elif optim == 'LBFGS':
        optimizer = torch.optim.LBFGS(dummies, lr=lr)

    for i in range(participants):
        for j in range(batch_size):
            history[i][j].append(dummy_datas[i][j].cpu().clone())
            if procedure_save is True:
                recover_procedure.append(dummy_datas[i][j].cpu().clone())

    start_time = time.time()
    if optim == 'LBFGS':
        for iter_num in range(iters):
            def closure():
                # compute mean dummy dy/dx
                total_dy_dx = []
                optimizer.zero_grad()
                smooth = 0
                for i in range(participants):
                    pred = net(dummy_datas[i])
                    dummy_loss = criterion(pred, dummy_labels[i])
                    dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                    # share the gradients with other clients
                    dummy_dy_dx = [_ for _ in dy_dx]
                    for j in range(len(dummy_dy_dx)):
                        if len(total_dy_dx) <= j:
                            total_dy_dx.append(dummy_dy_dx[j])
                        else:
                            total_dy_dx[j] += dummy_dy_dx[j]

                dummy_mean_dy_dx = []
                for i in range(len(total_dy_dx)):
                    dummy_mean_dy_dx.append(total_dy_dx[i] / participants)

                grad_diff = 0
                for gx, gy in zip(dummy_mean_dy_dx, mean_dy_dx):  
                    grad_diff += ((gx - gy) ** 2).sum()

                if config['norm_method'] == 'smooth':
                    for i in range(participants):
                        for j in range(batch_size):
                            smooth += compute_smooth_by_martix(dummy_datas[i][j], smooth_direction)
                    grad_diff += norm_rate * smooth

                grad_diff.backward()
                return grad_diff

            optimizer.step(closure)
            current_loss = closure()
            loss.append(current_loss.item())

            mean_psnr = calculate_psnr(dummy_datas[0].cpu().clone().detach(), gt_data[0].cpu().clone().detach())
            psnrs.append(mean_psnr)

            if procedure_save is True:
                for i in range(participants):
                    for j in range(batch_size):
                        recover_procedure[i][j].append(dummy_datas[i][j].cpu().clone())

            if (iter_num % step_size == 0) or iter_num == iters - 1:
                for i in range(participants):
                    for j in range(batch_size):
                        history[i][j].append(dummy_datas[i][j].cpu().clone())
                print("iter_num:{}\tloss:{:.5f}\tmean_psnr:{:.5f}\tcost time:{:.2f} secs".format(iter_num,
                                                                                                 current_loss.item(),
                                                                                                 mean_psnr,
                                                                                                 time.time() - start_time))
                start_time = time.time()

            if early_stop(psnrs, config, iter_num):
                break

        for i in range(participants):
            for j in range(batch_size):
                history[i][j].append(dummy_datas[i][j].cpu().clone())
                if procedure_save is True:
                    recover_procedure.append(dummy_datas[i][j].cpu().clone())

    elif optim == 'adam':
        for iter_num in range(iters):
            total_dy_dx = []
            optimizer.zero_grad()
            smooth = 0
            for i in range(participants):
                pred = net(dummy_datas[i])
                dummy_loss = criterion(pred, dummy_labels[i])
                dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                # share the gradients with other clients
                dummy_dy_dx = [_ for _ in dy_dx]
                for j in range(len(dummy_dy_dx)):
                    if len(total_dy_dx) <= j:
                        total_dy_dx.append(dummy_dy_dx[j])
                    else:
                        total_dy_dx[j] += dummy_dy_dx[j]

            dummy_mean_dy_dx = []
            for i in range(len(total_dy_dx)):
                dummy_mean_dy_dx.append(total_dy_dx[i] / participants)

            grad_diff = 0
            for gx, gy in zip(dummy_mean_dy_dx, mean_dy_dx):  
                grad_diff += ((gx - gy) ** 2).sum()

            if config['norm_method'] == 'smooth':
                # 图片smooth程度正则项
                for i in range(participants):
                    for j in range(batch_size):
                        smooth += compute_smooth_by_martix(dummy_datas[i][j], smooth_direction)
                        # smooth += compute_mean(dummy_datas[i][j])
                        # print(smooth)
                grad_diff += norm_rate * smooth

            grad_diff.backward()

            optimizer.step()
            current_loss = grad_diff.item()
            loss.append(current_loss)

            mean_psnr = calculate_psnr(dummy_datas[0].cpu().clone().detach(), gt_data[0].cpu().clone().detach())
            psnrs.append(mean_psnr)

            if procedure_save is True:
                for i in range(participants):
                    for j in range(batch_size):
                        recover_procedure[i][j].append(dummy_datas[i][j].cpu().clone())

            if (iter_num % step_size == 0) or iter_num == iters - 1:
                for i in range(participants):
                    for j in range(batch_size):
                        history[i][j].append(dummy_datas[i][j].cpu().clone())
                print("iter_num:{}\tloss:{:.5f}\tmean_psnr:{:.5f}\tcost time:{:.2f} secs".format(iter_num, current_loss,
                                                                                                 mean_psnr,
                                                                                                 time.time() - start_time))
                start_time = time.time()

            if early_stop(psnrs, config, iter_num):
                break

        for i in range(participants):
            for j in range(batch_size):
                history[i][j].append(dummy_datas[i][j].cpu().clone())
                if procedure_save is True:
                    recover_procedure.append(dummy_datas[i][j].cpu().clone())

    return dummy_datas, dummy_labels, history, recover_procedure, loss[1:], psnrs


def create_plt(save_dir, config, gt_data, dummy_datas, dummy_labels, history, recover_procedure, loss, psnrs=None):
    participants = config['participants']
    batch_size = config['batch_size']
    iters = config['iters']
    step_size = config['step_size']
    procedure_save = config['procedure_save']

    row = math.ceil(iters / step_size / 10)
    compare_result = []
    compare_truth = []
    history_grid = []

    if procedure_save is True:
        procedure_folder = os.path.join(save_dir, 'procedure')
        check_folder_path([procedure_folder])

    for p in range(participants):
        for j in range(batch_size):
            history_grid = []
            save_tensor_img(save_dir, 'result{}-{}'.format(p + 1, j + 1), dummy_datas[p][j].cpu())
            plt.figure(figsize=(20, 6))
            print("Participant {} Dummy label {} is {}.".format(p + 1, j + 1,
                                                                torch.argmax(dummy_labels[p][j], dim=-1).item()))
            for i in range(min(len(history[p][j]) - 1, (int)(iters / step_size))):
                history_grid.append(history[p][j][i])
            history_grid.append(history[p][j][-1])
            compare_result.append(dummy_datas[p][j])
            compare_truth.append(gt_data[p][j])
            save_tensor_img(save_dir, 'procedure{}-{}'.format(p + 1, j + 1),
                            history_grid, True,
                            dict(nrow=math.ceil(math.sqrt(len(history_grid)))))

            if procedure_save is True:
                for count in range(len(recover_procedure[p][j])):
                    save_tensor_img(procedure_folder, 'procedure{}-{}-{}'.format(p + 1, j + 1, count + 1),
                                    recover_procedure[p][j][count])

        save_tensor_img(save_dir, 'result_img_grid', dummy_datas[p], True)
        save_tensor_img(save_dir, 'compare',
                        [vutils.make_grid(dummy_datas[p], len(dummy_datas[p])),
                         vutils.make_grid(gt_data[p], len(gt_data[p]))],
                        True, dict(nrow=1))

    plt.figure(figsize=(12, 6))
    loss_ = [loss[i] for i in range(len(loss)) if i % step_size == 0]
    x = [i + 1 for i in range(len(loss)) if i % step_size == 0]
    plt.plot(x, loss_, color='#000000', label='loss')
    plt.title('Loss(log)')
    plt.xlabel('iter')
    plt.yscale('log')
    plt.ylabel('loss value')
    plt.xticks(range(len(x)))
    x_major_locator = MultipleLocator((int)(config['iters'] / 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    save_plt_img(save_dir, 'loss')

    plt.figure(figsize=(12, 6))
    psnrs_ = [psnrs[i] for i in range(len(psnrs)) if i % step_size == 0]
    x = [i + 1 for i in range(len(psnrs)) if i % step_size == 0]
    plt.plot(x, psnrs_, color='#000000', label='psnr')
    plt.title('Mean Psnr')
    plt.xlabel('iter')
    plt.ylabel('mean psnr value')
    plt.xticks(range(len(x)))
    x_major_locator = MultipleLocator((int)(config['iters'] / 10))
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.legend()
    save_plt_img(save_dir, 'meanpsnr')

    loss_log_path = os.path.join(save_dir, 'loss.log')
    loss_str_list = ['iter-{}:{:.5f}'.format(i + 1, loss[i]) for i in range(len(loss))]
    with open(loss_log_path, 'w') as f:
        f.write('\n'.join(loss_str_list))

    psnr_log_path = os.path.join(save_dir, 'meanpsnr.log')
    psnr_str_list = ['iter-{}:{:.5f}'.format(i + 1, psnrs[i]) for i in range(len(psnrs))]
    with open(psnr_log_path, 'w') as f:
        f.write('\n'.join(psnr_str_list))


def experiment_config_loop(mode, ckpt_location, context, experiments, current_config_idx, config_name):
    experiments[current_config_idx] = (experiments[current_config_idx] + 1) % len(
        experiments[config_name])
    context['experiments'] = experiments
    save_checkpoint(mode, ckpt_location, context)


def experiment(mode, device, experiments, config, base_generate_model_path, **idx):
    # 早停参数复原
    config['iters'] = experiments['iters']
    config['step_size'] = experiments['step_size']
    config['early_stop'] = False

    print('''
========================================================
Mode: {}
Batch Size: {}
Training Num: {}
Optimizer: {}
Learning Rate: {}
Init Method: {}
Dataset: {}
Norm Method: {}
Norm Rate: {}
Iters: {}
Step Size: {}
Smooth Direction: {}
Noise Type: {}
Noise Variance: {}
    '''.format(
        mode,
        experiments['batch_size'][idx['b_idx']],
        experiments['training_num'][idx['t_idx']],
        experiments['optim'],
        experiments['lr'][idx['l_idx']],
        experiments['init'][idx['init_idx']],
        experiments['dataset'][idx['ds_idx']].upper(),
        experiments['norm_methods'][idx['nm_idx']],
        experiments['norm_rate'][idx['nr_idx']],
        config['iters'],
        config['step_size'],
        experiments['smooth_direction'][idx['sd_idx']],
        experiments['noise_type'][idx['nt_idx']],
        experiments['noise_variance'][idx['nv_idx']]
    ))

    start_time = time.time()
    config['batch_size'] = experiments['batch_size'][idx['b_idx']]
    config['lr'] = experiments['lr'][idx['l_idx']]
    config['optim'] = experiments['optim']
    config['init_method'] = experiments['init'][idx['init_idx']]
    config['dataset'] = experiments['dataset'][idx['ds_idx']]
    config['norm_rate'] = experiments['norm_rate'][idx['nr_idx']]
    config['norm_method'] = experiments['norm_methods'][idx['nm_idx']]
    config['smooth_direction'] = experiments['smooth_direction'][idx['sd_idx']]
    config['noise_type'] = experiments['noise_type'][idx['nt_idx']]
    config['noise_variance'] = experiments['noise_variance'][idx['nv_idx']]

    generate_models = []
    generate_model = None
    if config['init_method'] == 'gan' and config['dataset'] == 'mnist':
        generate_model_path = os.path.join(base_generate_model_path,
                                           'MNIST-GenerateModel')

        generate_model = MnistGenerator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'modelG'),
            map_location=torch.device(device)))
        generate_model.to(device)
        generate_model.eval()
    elif config['init_method'] == 'gan' and config['dataset'] == 'cifar10':
        generate_model_path = os.path.join(base_generate_model_path,
                                           'CIFAR10-GenerateModel')
        generate_model = Cifar10Generator()
        generate_model.load_state_dict(
            torch.load(os.path.join(generate_model_path, 'modelG'),
                       map_location=torch.device(device)))
        generate_model.to(device)
        generate_model.eval()
    elif config['init_method'] == 'gan' and config['dataset'] == 'svhn':
        generate_model_path = os.path.join(base_generate_model_path,
                                           'SVHN-GenerateModel')
        generate_model = SVHNGenerator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'modelG'),
            map_location=torch.device(device)))
        generate_model.to(device)
        generate_model.eval()
    elif config['init_method'] == 'gan' and config['dataset'] == 'cifar100':
        generate_model_path = os.path.join(base_generate_model_path,
                                           'CIFAR100-GenerateModel')
        generate_model = CIFAR100Generator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'modelG'),
            map_location=torch.device(device)))
        generate_model.to(device)
        generate_model.eval()
    elif config['init_method'] == 'gan' and config['dataset'] == 'lfw':
        generate_model_path = os.path.join(base_generate_model_path,
                                           'LFW-GenerateModel')
        generate_model = LFWGenerator()
        generate_model.load_state_dict(torch.load(
            os.path.join(generate_model_path, 'modelG'),
            map_location=torch.device(device)))
        generate_model.to(device)
        generate_model.eval()

    net = None
    if config['dataset'] == 'mnist':
        config['num_classes'] = 10
        net = LeNet_Mnist().to(device)
    elif config['dataset'] == 'cifar10':
        config['num_classes'] = 10
        net = LeNet_Cifar10().to(device)
    elif config['dataset'] == 'cifar100':
        config['num_classes'] = 100
        net = LeNet_Cifar100().to(device)
    elif config['dataset'] == 'svhn':
        config['num_classes'] = 10
        net = LeNet_SVHN().to(device)
    elif config['dataset'] == 'lfw':
        config['num_classes'] = 5749
        net = LeNet_LFW().to(device)
    net.apply(weights_init)

    save_dir = get_save_path(experiments['training_num'][idx['t_idx']], config)
    gt_data, recoverd_onehot_label, mean_dy_dx, data_shape, label_onehot_shape = get_real_datas(
        net, save_dir, config)
    dummy_datas, dummy_labels = generate_dummy_datas(save_dir, config,
                                                     data_shape,
                                                     label_onehot_shape,
                                                     generate_model,
                                                     generate_models,
                                                     recoverd_onehot_label)
    # add noise
    noise_distributions = {
        'gaussian': torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor(
            [config['noise_variance']])) if config['noise_type'] != 'none' else None,
        'laplace': torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor(
            [config['noise_variance']])) if config['noise_type'] != 'none' else None,
        'none': None,
    }
    noise_distribution = noise_distributions[config['noise_type']]
    if noise_distribution is not None:
        with torch.no_grad():
            for p in net.parameters():
                p.data = p.data + noise_distribution.sample(p.data.size()).squeeze(-1).to(device)

    dummy_datas, dummy_labels, history, recover_procedure, loss, psnrs = recover(save_dir, config,
                                                                                 net, gt_data, dummy_datas,
                                                                                 recoverd_onehot_label,
                                                                                 mean_dy_dx)
    create_plt(save_dir, config, gt_data, dummy_datas, dummy_labels, history, recover_procedure,
               loss, psnrs)
    plt.close('all')
    cost_time = time.time() - start_time
    print('\ntime cost: {} secs'.format(cost_time))
    print(f'Experiments results saved in {os.path.abspath(save_dir)}')
    print('========================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN combine DLG')
    parser.add_argument('--config-file', type=str, help='experiment config file path')
    parser.add_argument('--mode', type=str, default='debug', help='experiment mode')
    parser.add_argument('--base-config', type=str, default='../base_config.json')
    args = parser.parse_args()

    base_config = read_experiment_config(args.base_config)
    folder_paths = [base_config['production_path'], base_config['debug_path'], base_config['generate_model_path']]
    check_folder_path(folder_paths)

    try:
        experiment_config = read_experiment_config(args.config_file)
    except (FileNotFoundError, TypeError):
        experiment_config = {}
    experiment_name = experiment_config.get('name', 'filterDLG')
    participants = experiment_config.get('participants', [1])
    batch_size = experiment_config.get('batch_size', [1])
    data_set = experiment_config.get('data_set', ['cifar100'])
    training_num = experiment_config.get('training_num', 2)
    init_methods = experiment_config.get('init_methods', ['gan'])
    norm_methods = experiment_config.get('norm_methods', ['smooth'])
    norm_rate = experiment_config.get('norm_rate', [1e-4])
    iters = experiment_config.get('iters', 1000)
    optim = experiment_config.get('optim', 'LBFGS')
    device = experiment_config.get('device', 'cpu')
    step_size = experiment_config.get('step_size', 1 if iters <= 100 else math.ceil(iters / 100))
    lr = experiment_config.get('learning_rate', [1])
    early_stop_step = experiment_config.get('early_stop_step', int(iters / 50))
    smooth_direction = experiment_config.get('smooth_direction', [4])
    procedure_save = experiment_config.get('procedure_save', False)
    truth_imgs = experiment_config.get('truth_imgs', None)
    noise_type = experiment_config.get('noise_type', ['none'])
    noise_variance = experiment_config.get('noise_variance', [0])

    if truth_imgs != None:
        assert len(truth_imgs) >= max(batch_size)
    mode = args.mode

    if torch.cuda.is_available() and device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    print("Running on %s" % device)

    assert participants == [1]

    production_path = base_config['production_path']
    debug_path = base_config['debug_path']
    base_generate_model_path = base_config['generate_model_path']
    path = debug_path

    if mode == 'production':
        path = production_path
    path = os.path.join(path, experiment_name)

    config = {
        'participants': 1,
        'batch_size': 1,
        'dataset': 'mnist',
        'lr': lr,
        'optim': 'LBFGS',
        'iters': iters,
        'step_size': step_size,
        'dir': path,
        'init_method': 'gan',
        'norm_method': 'none',
        'norm_rate': 0.0001,
        'regular_ratio': 0,
        'early_stop_step': early_stop_step,
        'procedure_save': procedure_save,
        'truth_imgs': truth_imgs,
        'noise_type': noise_type,
        'noise_variance': noise_variance
    }

    criterion = cross_entropy_for_onehot
    experiments = {
        'participants': participants,
        'current_participant': 0,
        'batch_size': batch_size,
        'current_bs': 0,
        'dataset': data_set,
        'current_ds': 0,
        'training_num': [i for i in range(1, training_num + 1)],
        'current_tn': 0,
        'optim': optim,
        'current_opt': 0,
        'lr': lr,
        'current_lr': 0,
        'norm_rate': norm_rate,
        'current_nr': 0,
        'init': init_methods,
        'current_init': 0,
        'norm_methods': norm_methods,
        'current_nm': 0,
        'iters': iters,
        'step_size': step_size,
        'smooth_direction': smooth_direction,
        'current_sd': 0,
        'noise_type': noise_type,
        'current_nt': 0,
        'noise_variance': noise_variance,
        'current_nv': 0
    }

    done = False
    ckpt_location = os.path.join(path, 'context.ckpt')
    context = load_checkpoint(ckpt_location)
    if context is None:
        context = {
            'done': False,
            'config': config,
            'experiments': experiments
        }
        print('init experiments context')
    elif context['done'] == False:
        config = context['config']
        experiments = context['experiments']
        print('load unfinished experiment')
        print(experiments)
    elif context['done'] == True:
        print('All experiments are done!')
        done = True

    if not done:
        current_participant = experiments['current_participant']
        current_bs = experiments['current_bs']
        current_tn = experiments['current_tn']
        current_lr = experiments['current_lr']
        current_init = experiments['current_init']
        current_ds = experiments['current_ds']
        current_nr = experiments['current_nr']
        current_nm = experiments['current_nm']
        current_sd = experiments['current_sd']
        current_nt = experiments['current_nt']
        current_nv = experiments['current_nv']

        for b_idx in range(current_bs, len(experiments['batch_size'])):
            for t_idx in range(current_tn, len(experiments['training_num'])):
                for l_idx in range(current_lr, len(experiments['lr'])):
                    for init_idx in range(current_init, len(experiments['init'])):
                        for ds_idx in range(current_ds, len(experiments['dataset'])):
                            for nr_idx in range(current_nr, len(experiments['norm_rate'])):
                                for nm_idx in range(current_nm, len(experiments['norm_methods'])):
                                    for sd_idx in range(current_sd, len(experiments['smooth_direction'])):
                                        for nt_idx in range(current_nt, len(experiments['noise_type'])):
                                            for nv_idx in range(current_nv, len(experiments['noise_variance'])):
                                                idx = dict(b_idx=b_idx, t_idx=t_idx,
                                                           l_idx=l_idx, init_idx=init_idx, ds_idx=ds_idx, nr_idx=nr_idx,
                                                           nm_idx=nm_idx, sd_idx=sd_idx, nt_idx=nt_idx, nv_idx=nv_idx)
                                                experiment(mode, device, experiments, config, base_generate_model_path,
                                                           **idx)
                                                experiment_config_loop(mode, ckpt_location, context, experiments,
                                                                       'current_nv', 'noise_variance')
                                            experiment_config_loop(mode, ckpt_location, context, experiments,
                                                                   'current_nt', 'noise_type')
                                        experiment_config_loop(mode, ckpt_location, context, experiments, 'current_sd',
                                                               'smooth_direction')
                                    experiment_config_loop(mode, ckpt_location, context, experiments, 'current_nm',
                                                           'norm_methods')
                                experiment_config_loop(mode, ckpt_location, context, experiments, 'current_nr',
                                                       'norm_rate')
                            experiment_config_loop(mode, ckpt_location, context, experiments, 'current_ds', 'dataset')
                        experiment_config_loop(mode, ckpt_location, context, experiments, 'current_init', 'init')
                    experiment_config_loop(mode, ckpt_location, context, experiments, 'current_lr', 'lr')
                experiment_config_loop(mode, ckpt_location, context, experiments, 'current_tn', 'training_num')
            experiment_config_loop(mode, ckpt_location, context, experiments, 'current_bs', 'batch_size')

    context['done'] = True
    save_checkpoint(mode, ckpt_location, context)
    print('All Expriments Done!')
