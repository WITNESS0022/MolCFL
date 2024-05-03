import os
import argparse
import torch
from trainer import Trainer
from Dataloader import get_loader
from torch.backends import cudnn
from tqdm import tqdm
from molecular_dataset import MolecularDataset
import numpy as np
import copy
# from utils import average_weights
import utils
import matplotlib.pyplot as plt
import json
import sys

import Similarity_Analysis

def str2bool(v):
    return v.lower() in ('true')


def main(args):
    cudnn.benchmark = True

    # Since graph input sizes remains constant
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    data = MolecularDataset()
    data.load(args.mol_data_dir)
    idxs = len(str(data))

    # trainer for training and testing StarGAN.
    train_data, test_data, user_groups = get_loader(args)
    trainer = Trainer(args, data, idxs)

    g_global_model, d_global_model = trainer.build_model()

    # global model weights
    g_global_weights = g_global_model.state_dict()
    d_global_weights = d_global_model.state_dict()

    g_train_loss, d_train_loss = [], []
    g_last_local_loss, d_last_local_loss = [], []

    idxs_users_list =[]

    log_path = "/workspace/algorithm/cross_esol/logs/cluster_log.txt"
    # Open log file
    log_file = open(log_path, "a")

    # Save original stdout
    original_stdout = sys.stdout

    # Redirect stdout to log file
    sys.stdout = log_file

    cudnn.benchmark = True

    if args.mode == 'train':
        for i in tqdm(range(args.epochs_global)):
            if i == 0:
                print('Round 0')
                g_local_weights, g_local_losses, d_local_weights, d_local_losses = [], [], [], []

                # g_last_local_loss.clear()
                # d_last_local_loss.clear()

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = np.random.choice(range(args.num_users), m, replace=False)

                for idx in idxs_users:
                    local_model = Trainer(args=args, data=train_data, idxs=user_groups[idx])
                    g_weights, d_weights, g_loss, d_loss = local_model.tnr(model=copy.deepcopy(d_global_model),
                                                                           global_round=i)


                    g_local_weights.append(copy.deepcopy(g_weights))
                    g_local_losses.append(copy.deepcopy(g_loss))

                    d_local_weights.append(copy.deepcopy(d_weights))
                    d_local_losses.append(copy.deepcopy(d_loss))

                    g_last_local_loss.append(g_local_losses[-1])
                    d_last_local_loss.append(d_local_losses[-1])

                g_local_losses = np.array(g_last_local_loss).ravel()
                d_local_losses = np.array(d_last_local_loss).ravel()

                # average local weights
                g_global_weights = utils.average_weights(g_local_weights)
                d_global_weights = utils.average_weights(d_local_weights)

                # remove_prefix = "module."
                # g_global_weights = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in g_global_weights.items()}
                # d_global_weights = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in d_global_weights.items()}

                # update global weights
                g_global_model.load_state_dict(g_global_weights)
                d_global_model.load_state_dict(d_global_weights)

                g_loss_avg = sum(g_local_losses) / len(g_local_losses)
                d_loss_avg = sum(d_local_losses) / len(d_local_losses)

                g_train_loss.append(g_loss_avg)
                d_train_loss.append(d_loss_avg)

                idxs_users = [idxs_users]
                g_local_losses, d_local_losses = [], []


            else:
                print(f'\n | Global Training Round : {i + 1} |\n')
                # Calculate the similarity matrix of model updates between all clients
                g_similarities = Similarity_Analysis.compute_pairwise_similarities(g_local_weights)
                d_similarities = Similarity_Analysis.compute_pairwise_similarities(d_local_weights)

                idxs_users_new = []
                for idx in idxs_users:
                    # Calculate the maximum norm and average norm of the weight updates of the local generator and discriminator
                    g_max_norm = Similarity_Analysis.compute_max_update_norm(g_local_weights)
                    g_mean_norm = Similarity_Analysis.compute_mean_update_norm(g_local_weights)

                    d_max_norm = Similarity_Analysis.compute_max_update_norm(d_local_weights)
                    d_mean_norm = Similarity_Analysis.compute_mean_update_norm(d_local_weights)

                    if g_mean_norm < args.EPS_1 and g_max_norm > args.EPS_2 and len(idx) > 2 and i > 1:
                        c1, c2 = Similarity_Analysis.cluster_clients(g_similarities[idx][:, idx])
                        c1_mapped = [idx[i] for i in c1]
                        c2_mapped = [idx[i] for i in c2]
                        idxs_users_new += [c1_mapped, c2_mapped]

                    else:
                        idxs_users_new += [idx]

                idxs_users = idxs_users_new
                idxs_users_list.append(idxs_users)

                # local training
                for cluster in idxs_users:
                    for idx in cluster:
                        local_model = Trainer(args=args, data=train_data, idxs=user_groups[idx])
                        g_weights, d_weights, g_loss, d_loss = local_model.tnr(model=copy.deepcopy(d_global_model),
                                                                               global_round=i)

                        g_local_weights.append(copy.deepcopy(g_weights))
                        g_local_losses.append(copy.deepcopy(g_loss))

                        d_local_weights.append(copy.deepcopy(d_weights))
                        d_local_losses.append(copy.deepcopy(d_loss))

                        g_last_local_loss.append(g_local_losses[-1])
                        d_last_local_loss.append(d_local_losses[-1])

                    # intra-group aggregation
                    # average cluster weights
                    g_cluster_weights = utils.average_weights(g_local_weights)
                    d_cluster_weights = utils.average_weights(d_local_weights)

                # global updating
                # average cluster weights
                g_global_weights = utils.average_weights([g_cluster_weights])
                d_global_weights = utils.average_weights([d_cluster_weights])

                # update global weights
                g_global_model.load_state_dict(g_global_weights)
                d_global_model.load_state_dict(d_global_weights)

                g_local_losses_np = np.array(g_last_local_loss).ravel()
                d_local_losses_np = np.array(d_last_local_loss).ravel()

                g_loss_avg = sum(g_local_losses_np) / len(g_local_losses_np)
                d_loss_avg = sum(d_local_losses_np) / len(d_local_losses_np)

                g_train_loss.append(g_loss_avg)
                d_train_loss.append(d_loss_avg)


        plt.figure(figsize=(10, 5))
        plt.plot(range(args.epochs_global), g_train_loss, label="Generator")
        plt.plot(range(args.epochs_global), d_train_loss, label="Discriminator")
        plt.xlabel("Global rounds", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.xticks(np.arange(0, args.epochs_global, 2), fontsize=12)
        plt.title("Generator and Discriminator Loss for FedAvg")
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig('/workspace/algorithm/cross_esol/logs/Generator and Discriminator Loss for FedAvg.pdf')
        plt.show()

    elif args.mode == 'test':
        trainer.test()

    # Restore original stdout
    sys.stdout = original_stdout

    # Close log file
    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Norm criteria.
    parser.add_argument('--EPS_1', type=int, default=None, help='the first epsilon value')
    parser.add_argument('--EPS_2', type=int, default=None, help='the second epsilon value')

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=16, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim', default=[16, 32, 64], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[32, 64], 32, [64, 1]],
                        help='number of conv filters in the first layer of D')  # [128, 64], 128, [128, 64]
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='softmax', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')  # 16
    parser.add_argument('--num_iters_local', type=int, default=5000,
                        help='number of total iterations for training D')  # 200000
    parser.add_argument('--num_iters_decay', type=int, default=10,
                        help='number of iterations for decaying lr')  # 100000
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--epochs_global', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help='the fraction of clients: C')

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=5000, help='test model from this step')  # 200000

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--data_iid', type=int, default=0, help='Default set to IID. Set to 0 for non-IID.')
    # parser.add_argument('--data_noniid', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')

    # Directories.
    parser.add_argument('--mol_data_dir', type=str,
                        default='/workspace/algorithm/cross_esol/data-smiles/esol.dataset')
    parser.add_argument('--log_dir', type=str, default='/workspace/algorithm/cross_esol/logs')
    parser.add_argument('--model_save_dir', type=str, default='/workspace/algorithm/cross_esol/results/esol_k=5_glo=10_[16,32,64],[[32,64],32,[64,1]]/models')
    # parser.add_argument('--model_save_dir', type=str, default='/workspace/algorithm/cross_esol/results1/esol_k=5_glo=50_[32,64,[[32,64],32,[64,1,]]/models1')
    parser.add_argument('--sample_dir', type=str, default='/workspace/algorithm/cross_esol/samples')
    parser.add_argument('--result_dir', type=str, default='/workspace/algorithm/cross_esol/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)  # 10
    parser.add_argument('--sample_step', type=int, default=1000)  # 1000
    parser.add_argument('--model_save_step', type=int, default=1000)  # 10000
    parser.add_argument('--lr_update_step', type=int, default=1000)  # 1000

    args = parser.parse_args()
    print(args)
    main(args)
