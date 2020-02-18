from config import get_config
from Learner_corr import face_learner as face_learner_corr
from Learner import face_learner
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=20, type=int)
    parser.add_argument("-net", "--net_mode", help="networks: [ir, ir_se, mobilefacenet]", default='ir_se', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="number of workers", default=3, type=int)
    parser.add_argument("-d", "--data_mode", help="databases: [vgg, ms1m, emore, concat]", default='emore', type=str)

    # ganovich - added some parameters
    parser.add_argument("-n", "--n_models", help="how many duplicate nets to use. 1 leads to basic training, "
                                                 "making -a and -p flags redundant", default=1, type=int)
    # TODO maybe add option to specify a network mix instead of duplicates
    parser.add_argument("-m", "--milestones", help="epoch list where lr will be tuned", default=[12, 15, 18], type=list)
    parser.add_argument("-a", "--alpha", help="balancing parameter", default=0, type=float)
    parser.add_argument("-p", "--pearson", help="using pearson loss", default=False, type=bool)
    parser.add_argument("-mean", "--joint_mean", help="using mean loss", default=False, type=bool)
    parser.add_argument("-c", "--cpu_mode", help="force cpu mode", default=False, type=bool)

    args = parser.parse_args()
    conf = get_config()

    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth

    # ganovich - added some parameters
    conf.milestones = args.milestones
    conf.alpha = args.alpha
    conf.n_models = args.n_models
    conf.n_models = args.n_models
    conf.pearson = args.pearson
    conf.joint_mean = args.joint_mean
    conf.cpu_mode = args.cpu_mode
    conf.device = torch.device("cuda:0" if (torch.cuda.is_available() and not conf.cpu_mode) else "cpu")

    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    learner = face_learner_corr(conf)  # face_learner(conf) if conf.n_models == 1 else face_learner_corr(conf)

    learner.train(conf, args.epochs)
