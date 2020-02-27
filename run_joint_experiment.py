import argparse, os, subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-m_dir", "--model_dir", type=str)
    args = parser.parse_args()

    models = [os.path.join(args.model_dir, x) for x in os.listdir(args.model_dir) if 'model' in x]
    for model in models:
        subprocess.call(, shell=True)