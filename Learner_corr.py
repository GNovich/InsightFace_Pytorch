from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate as ver_evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
plt.switch_backend('agg')


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)

        if conf.use_mobilfacenet:
            self.models = [MobileFaceNet(conf.embedding_size).to(conf.device) for _ in range(conf.n_models)]
            print('MobileFaceNet models generated')
        else:
            self.models = [Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device) for _ in
                           range(conf.n_models)]
            print('{}_{} models generated'.format(conf.net_mode, conf.net_depth))

        self.inference = inference
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.heads = [Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)
                          for _ in range(conf.n_models)]

            print('two model heads generated')

            paras_only_bn = []
            paras_wo_bn = []
            for model in self.models:
                paras_only_bn_, paras_wo_bn_ = separate_bn_paras(model)
                paras_only_bn.append(paras_only_bn_)
                paras_wo_bn.append(paras_wo_bn_)

            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                               {'params': paras_only_bn[model_num]}
                                               for model_num in range(conf.n_models)
                                           ] + [
                                               {'params': paras_wo_bn[model_num][:-1], 'weight_decay': 4e-5}
                                               for model_num in range(conf.n_models)
                                           ] + [
                                               {'params': [paras_wo_bn[head_num][-1]] + [self.heads[head_num].kernel],
                                                'weight_decay': 4e-4}
                                               for head_num in range(conf.n_models)
                                           ], lr=conf.lr, momentum=conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                               {'params': paras_wo_bn[head_num] + [self.heads[head_num].kernel],
                                                'weight_decay': 5e-4}
                                               for head_num in range(conf.n_models)
                                           ] + [
                                               {'params': paras_only_bn[model_num]}
                                               for model_num in range(conf.n_models)
                                           ], lr=conf.lr, momentum=conf.momentum)
            print(self.optimizer)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')
            self.board_loss_every = len(self.loader) // 100
            self.evaluate_every = len(self.loader) // 10
            self.save_every = len(self.loader) // conf.save_per_epoch
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(
                self.loader.dataset.root.parent)
        else:
            self.threshold = conf.threshold

    def save_state(self, conf, accuracy, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        for mod_num in range(conf.n_models):
            torch.save(
                self.models[mod_num].state_dict(), save_path /
                                                   ('model_{}_{}_accuracy:{}_step:{}_{}.pth'.format(mod_num, get_time(),
                                                                                                    accuracy, self.step,
                                                                                                    extra)))
        if not model_only:
            for mod_num in range(conf.n_models):
                torch.save(
                    self.heads[mod_num].state_dict(), save_path /
                                                     ('head_{}_{}_accuracy:{}_step:{}_{}.pth'.format(mod_num,
                                                                                                     get_time(),
                                                                                                     accuracy,
                                                                                                     self.step,
                                                                                                     extra)))
            torch.save(
                self.optimizer.state_dict(), save_path /
                                             ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy,
                                                                                               self.step, extra)))

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        def load_fix(target_path):
            a = torch.load(target_path)
            fixed_a = {k.split('module.')[-1]: a[k] for k in a}
            torch.save(fixed_a, target_path)

        for mod_num in range(conf.n_models):
            target_path = save_path / 'model_{}_{}'.format(mod_num, fixed_str)
            load_fix(target_path)
            self.models[mod_num].load_state_dict(torch.load(target_path))
        if not model_only:
            for mod_num in range(conf.n_models):
                target_path = save_path / 'head_{}_{}'.format(mod_num, fixed_str)
                load_fix(target_path)
                self.heads[mod_num].load_state_dict(torch.load(target_path))
            target_path = save_path / 'optimizer_{}'.format(fixed_str)
            load_fix(target_path)
            self.optimizer.load_state_dict(torch.load(target_path))

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)

    #   self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
    #   self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
    #   self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)

    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False, model_num=0):
        model = self.models[model_num]
        model.eval()

        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = model(batch.to(conf.device)) + model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = model(batch.to(conf.device)) + model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = ver_evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def find_lr(self, conf, init_value=1e-8, final_value=10., beta=0.98, bloding_scale=3., num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr

        for model in self.models:
            model.train()

        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)

            batch_num += 1
            self.optimizer.zero_grad()

            # calc embeddings
            thetas = []
            joint_losses = []
            for model, head in zip(self.models, self.heads):
                theta = head(model(imgs), labels)
                thetas.append(theta)
                joint_losses.append(conf.ce_loss(theta, labels))
            joint_losses = sum(joint_losses) / len(joint_losses)

            # calc loss
            if conf.pearson:
                outputs = torch.stack(thetas)
                pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                alpha = conf.alpha
                loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
            elif conf.joint_mean:
                mean_output = torch.mean(torch.stack(thetas), 0)
                ensemble_loss = conf.ce_loss(mean_output, labels)
                alpha = conf.alpha
                loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
            else:
                loss = joint_losses

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss, batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            # Do the SGD step
            # Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses

    def train(self, conf, epochs):
        for model_num in range(conf.n_models):
            self.models[model_num].train()

            if conf.resume:
                if not conf.fixed_str:
                    raise ValueError('must input fixed_str parameter!')
                # note if bach_size has changed - step and epoch would not match
                # note incomplete epoch will restart
                self.load_state(conf, conf.fixed_str, from_save_folder=False)
                self.step = int(conf.fixed_str.split('_')[-2].split(':')[1]) + 1
                start_epoch = self.step // len(self.loader)
                self.step = start_epoch * len(self.loader) + 1
                print('loading model at epoch {} done!'.format(start_epoch))
                print(self.optimizer)

            # ganovich mult. gpu. fix. start. based on: https://github.com/TreB1eN/InsightFace_Pytorch/issues/32
            if not conf.cpu_mode:
                self.models[model_num] = torch.nn.DataParallel(self.models[model_num], device_ids=[0, 1, 2, 3])

            self.models[model_num].to(conf.device)
            # ganovich mult. gpu. fix. end.

        running_loss = 0.
        running_pearson_loss = 0.
        running_ensemble_loss = 0.
        epoch_iter = range(epochs) if not conf.resume else range(start_epoch, epochs)
        for e in epoch_iter:
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()
            if e == self.milestones[2]:
                self.schedule_lr()
            for imgs, labels in tqdm(self.loader):
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()

                # calc embeddings
                thetas = []
                joint_losses = []
                for model, head in zip(self.models, self.heads):
                    theta = head(model(imgs), labels)
                    thetas.append(theta)
                    joint_losses.append(conf.ce_loss(theta, labels))
                joint_losses = sum(joint_losses) / max(len(joint_losses), 1)

                # calc loss
                if conf.pearson:
                    outputs = torch.stack(thetas)
                    pearson_corr_models_loss = conf.pearson_loss(outputs, labels)
                    running_pearson_loss += pearson_corr_models_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses + alpha * pearson_corr_models_loss
                elif conf.joint_mean:
                    mean_output = torch.mean(torch.stack(thetas), 0)
                    ensemble_loss = conf.ce_loss(mean_output, labels)
                    running_ensemble_loss += ensemble_loss.item()
                    alpha = conf.alpha
                    loss = (1 - alpha) * joint_losses * 0.5 + alpha * ensemble_loss
                else:
                    loss = joint_losses

                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    running_loss = 0.

                    if conf.pearson:  # ganovich listening to pearson
                        loss_board = running_pearson_loss / self.board_loss_every
                        self.writer.add_scalar('pearson_loss', loss_board, self.step)
                        running_pearson_loss = 0.

                    if conf.joint_mean:
                        loss_board = running_ensemble_loss / self.board_loss_every
                        self.writer.add_scalar('ensemble_loss', loss_board, self.step)
                        running_ensemble_loss = 0.

                # ganovich - listening to many models
                for model_num in range(conf.n_models):
                    if self.step % self.evaluate_every == 0 and self.step != 0:
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30,
                                                                                   self.agedb_30_issame,
                                                                                   model_num=model_num)
                        self.board_val('mod_{}_agedb_30'.format(model_num), accuracy, best_threshold, roc_curve_tensor)
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame,
                                                                                   model_num=model_num)
                        self.board_val('mod_{}_lfw'.format(model_num), accuracy, best_threshold, roc_curve_tensor)
                        accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp,
                                                                                   self.cfp_fp_issame,
                                                                                   model_num=model_num)
                        self.board_val('mod_{}_cfp_fp'.format(model_num), accuracy, best_threshold, roc_curve_tensor)
                        self.models[model_num].train()
                    if self.step % self.save_every == 0 and self.step != 0:
                        self.save_state(conf, accuracy)

                self.step += 1

        self.save_state(conf, accuracy, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:
            params['lr'] /= 10
        print(self.optimizer)

    def infer(self, conf, faces, target_embs, tta=False, model_num=0):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        model = self.models[model_num]
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
