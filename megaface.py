import argparse
import json
import os
import struct
from Pearson import set_bn_eval
import cv2 as cv
import numpy as np
import torch
import tqdm
from PIL import Image
from tqdm import tqdm
from Arc_coll_config import device
from data_gen import data_transforms
from Arc_coll_utils import align_face, get_central_face_attributes


def walkdir(folder, ext):
    # Walk through each files in a directory
    for dirpath, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.lower().endswith(ext)]:
            yield os.path.abspath(os.path.join(dirpath, filename))


def crop_one_image(filepath, oldkey, newkey):
    new_fn = filepath.replace(oldkey, newkey)
    tardir = os.path.dirname(new_fn)
    if not os.path.isdir(tardir):
        os.makedirs(tardir)

    if not os.path.exists(new_fn):
        is_valid, bounding_boxes, landmarks = get_central_face_attributes(filepath)
        if is_valid:
            img = align_face(filepath, landmarks)
            cv.imwrite(new_fn, img)


def crop(path, oldkey, newkey):
    print('Counting images under {}...'.format(path))
    # Preprocess the total files count
    filecounter = 0
    for filepath in walkdir(path, '.jpg'):
        filecounter += 1

    for filepath in tqdm(walkdir(path, '.jpg'), total=filecounter, unit="files"):
        crop_one_image(filepath, oldkey, newkey)

    print('{} images were cropped successfully.'.format(filecounter))


def gen_feature(path, models, batch_size=256, ext='.png', suffix='0'):
    print('gen features {}...'.format(path))
    # Preprocess the total files count
    files = []
    for filepath in walkdir(path, ext):
        files.append(filepath)
    file_count = len(files)

    with torch.no_grad():
        for start_idx in tqdm(range(0, file_count, batch_size)):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs = torch.zeros([length, 3, 112, 112], dtype=torch.float)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs[idx] = get_image(cv.imread(filepath, True), transformer)

            if len(models) == 1:
                features = models[0](imgs.to(device)).cpu().numpy()
            else:  # MEAN emmbeding
                features = (sum([model(imgs.to(device)) for model in models]) / len(models)).cpu().numpy()
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                tarfile = filepath + '_' + suffix + '.bin'
                feature = features[idx]
                write_feature(tarfile, feature / np.linalg.norm(feature))


def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def read_feature(filename):
    f = open(filename, 'rb')
    rows, cols, stride, type_ = struct.unpack('iiii', f.read(4 * 4))
    mat = np.fromstring(f.read(rows * 4), dtype=np.dtype('float32'))
    return mat.reshape(rows, 1)


def write_feature(filename, m):
    header = struct.pack('iiii', m.shape[0], 1, 4, 5)
    f = open(filename, 'wb')
    f.write(header)
    f.write(m.data)


def remove_noise(ext='0'):
    for line in open('data/megaface/megaface_noises.txt', 'r'):
        filename = 'data/megaface/megaface_images_aligned/' + line.strip() + '_' + ext +'.bin'
        if os.path.exists(filename):
            print(filename)
            os.remove(filename)

    for line in open('data/megaface/megaface_noises.txt', 'r'):
        filename = 'data/megaface/megaface_images/' + line.strip() + '_' + ext +'.bin'
        if os.path.exists(filename):
            print(filename)
            os.remove(filename)

    noise = set()
    for line in open('data/megaface/facescrub_noises.txt', 'r'):
        noise.add((line.strip() + ext + '.bin').replace('_', '').replace(' ', ''))
    for root, dirs, files in os.walk('data/megaface/facescrub_images'):
        for f in files:
            if f.replace('_', '').replace(' ', '') in noise:
                filename = os.path.join(root, f)
                if os.path.exists(filename):
                    print(filename)
                    os.remove(filename)


def test():
    root1 = 'data/FaceScrub_aligned/Benicio Del Toro'
    root2 = 'data/FaceScrub_aligned/Ben Kingsley'
    for f1 in os.listdir(root1):
        for f2 in os.listdir(root2):
            if f1.lower().endswith('.bin') and f2.lower().endswith('.bin'):
                filename1 = os.path.join(root1, f1)
                filename2 = os.path.join(root2, f2)
                fea1 = read_feature(filename1)
                fea2 = read_feature(filename2)
                print(((fea1 - fea2) ** 2).sum() ** 0.5)


def match_result():
    with open('matches_facescrub_megaface_0_1000000_1.json', 'r') as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
        for i in range(len(load_dict)):
            print(load_dict[i]['probes'])


def pngtojpg(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f)[1] == '.png':
                img = cv.imread(os.path.join(root, f))
                newfilename = f.replace(".png", ".jpg")
                cv.imwrite(os.path.join(root, newfilename), img)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--action', default='crop_megaface', help='action')
    parser.add_argument('--models', nargs='+', type=str)
    parser.add_argument('--suffix', default='0', help="names of emmbedding will be \'fname_suffix.bin\'", type=str)
    parser.add_argument('--batch_size', default=256, help="batch size", type=int)
    args = parser.parse_args()
    return args


def load_fix(target_path):
    a = torch.load(target_path)
    fixed_a = {k.split('module.')[-1]: a[k] for k in a}
    torch.save(fixed_a, target_path)


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'crop_megaface':
        crop('data/megaface/megaface_images', 'megaface_images', 'megaface_images_aligned')
    elif args.action == 'crop_facescrub':
        crop('data/megaface/facescrub_images', 'facescrub_images', 'facescrub_images_aligned')
    elif args.action == 'gen_features':
        models_path = args.models
        print('loading model: {}...'.format(models_path))
        from model import Backbone
        from config import get_config
        conf = get_config()
        models = []
        for model_path in models_path:
            model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(device)
            load_fix(model_path)
            model.load_state_dict(torch.load(model_path))
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
            if 'morph' in models_path:
                set_bn_eval(model)
            else:
                model.eval()
            models.append(model)
        transformer = data_transforms['ir_se50' if ('ir_se50' in models_path) else 'val']

        gen_feature('data/megaface/facescrub_images', models, batch_size=args.batch_size, ext='.png', suffix=args.suffix)
        # gen_feature('data/megaface/megaface_images_aligned', models, batch_size=args.batch_size, ext='.jpg', suffix=args.suffix)
        gen_feature('data/megaface/megaface_images', models, batch_size=args.batch_size, ext='.jpg', suffix=args.suffix)
        remove_noise(args.suffix)
    elif args.action == 'remove_noise':
        remove_noise(args.suffix)
    elif args.action == 'pngtojpg':
        pngtojpg('data/megaface/facescrub_images')
