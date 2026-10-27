import pathlib

from utils.args import Args


def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    parser.add_argument('--resume', action='store_true',
                        help='Whether or not to resume training.')
    parser.add_argument('--resume-epoch', default=0, type=int, help='Epoch to resume training from')
    parser.add_argument('--config', type=str, default='./configs/mri.yml', help='Path to the config file') 
    parser.add_argument('--exp-name', type=str, default="default_exp_name", help='Name for the run.')
    parser.add_argument('--num-gpus', default=1, type=int, help='The number of GPUs to use during training.')
    parser.add_argument('--num-figs', default=1, type=int, help='The number of figures to generate while plotting.')
    parser.add_argument('--crps-mode', type=str, default=None,
                        help='CRPS loss variant to use for mmGAN: "standard", "fair", "alpha_fair" or "ms". '
                             'Overrides the default hard-coded in mmGAN.__init__ when set.')
    parser.add_argument('--crps-alpha', type=float, default=None,
                        help='Alpha mixing parameter for the alpha_fair CRPS loss. '
                             'Overrides the default hard-coded in mmGAN.__init__ when set.')
    parser.add_argument('--adv-scale', type=float, default=1.0,
                        help='Multiplier on the generator adversarial loss for mmGAN (0 = no adversarial loss).')

    return parser
