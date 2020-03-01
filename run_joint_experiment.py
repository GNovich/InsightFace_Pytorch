import argparse
import os
from megaface import load_fix, gen_feature, remove_noise
from data_gen import data_transforms
from model import Backbone
from config import get_config
from Arc_coll_config import device
import torch
from megaface.devkit.experiments.run_experiment import main as run_experiment
from megaface.devkit.experiments.run_experiment import MEGAFACE_LIST_BASENAME, MODEL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-m_dir", "--model_dir", type=str)
    args = parser.parse_args()

    dirname = os.path.basename(os.path.dirname(args.model_dir))
    models_path = [os.path.join(args.model_dir, x) for x in os.listdir(args.model_dir) if 'model' in x]
    conf = get_config()

    # generate *bin
    models = []
    suffixes = []
    for model_path in models_path:
        suffix = dirname + '_' + models_path.split('_')[1]
        suffixes.append(suffix)

        model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
        load_fix(model_path)
        model.load_state_dict(torch.load(model_path))
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
        models.append(model)
        transformer = data_transforms['ir_se50' if ('ir_se50' in model_path) else 'val']

        gen_feature('data/megaface/facescrub_images', model, transformer, batch_size=args.batch_size, ext='.png', suffix=suffix)
        gen_feature('data/megaface/megaface_images', model, transformer, batch_size=args.batch_size, ext='.jpg', suffix=suffix)
        remove_noise()

        # generate results
        """  results/baseline_1 -s 1000000"""
        args = dict()
        args.probe_list = 'data/megaface/devkit/templatelists/facescrub_uncropped_features_list.json'
        args.distractor_list_path = MEGAFACE_LIST_BASENAME
        args.distractor_feature_path = 'data/megaface/megaface_images'
        args.probe_feature_path = 'data/megaface/facescrub_images'
        args.out_root = 'results/' + dirname + '/' + suffix
        args.model = os.path.dirname(MEGAFACE_LIST_BASENAME)
        args.num_sets = 1
        args.sizes = 1000000
        args.file_ending = '_' + suffix + '.bin'
        args.delete_matrices = False
        run_experiment(args)
