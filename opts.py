"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- data path -------------------------#
    parser.add_argument('--data_path', default='D:/pytorch/MMF/data/',
                        help='path to datasets')
    parser.add_argument('--data_name', default='f30k',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='D:/pytorch/MMF/vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='D:/pytorch/MMF/runs/f30k_SGR/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='D:/pytorch/MMF/runs/f30k_SGR/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--feature_path', default='D:/pytorch/MMF/data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/trainval/',
                        type=str, help='path to the pre-computed image features')
    parser.add_argument('--region_bbox_file',
                        default='D:/pytorch/MMF/data/joint-pretrain/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5',
                        type=str, help='path to the region_bbox_file(.h5)')

    # ----------------------- training setting ----------------------#
    parser.add_argument('--batch_size', default=24, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=5, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--learning_rate', default=.00005, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=100, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=5000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--resume', default='D:/pytorch/MMF/runs/f30k_SGR/checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_false',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--use_restval', action='store_true',
                        help='Use the restval data for training on MSCOCO.')

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')
    parser.add_argument('--K', default=2, type=int,
                        help='num of JSR.')

    opt = parser.parse_args()
    print(opt)
    return opt
