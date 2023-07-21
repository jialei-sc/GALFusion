import argparse


parser = argparse.ArgumentParser(description='MEF')

parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--dir_dataset', type=str, default='dataset/',
                    help='dataset directory')
parser.add_argument('--dir_train', type=str, default='train_data/',
                    help='train dataset name')
parser.add_argument('--dir_test', type=str, default='test_data/',
                    help='test dataset name')
parser.add_argument('--dir_val', type=str, default='val_data/',
                    help='val dataset name')
parser.add_argument('--model_pth', type=str, default='model/model_fuse.pth',
                    help='val dataset name')
parser.add_argument('--save_img', type=str, default='output_results/',
                    help='val dataset name')

parser.add_argument('--n_feats', type=int, default=128,
                    help='number of feature maps') #[128,256]
parser.add_argument('--res_scale', type=float, default=0.1,
                    help='residual scaling')
parser.add_argument('--lamda_mse', type=float, default=20,
                    help='loss hyparameters') #5
parser.add_argument('--lamda_ssim', type=float, default=1,
                    help='loss hyparameters')  #0.3
parser.add_argument('--lamda_tv', type=float, default=20,
                    help='loss hyparameters')
parser.add_argument('--n_resblocks', type=int, default=8,
                    help='number of residual blocks')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')



args = parser.parse_args()


