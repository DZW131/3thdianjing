import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', force_best=False):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best or force_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if force_best:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
                return
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        with open(logfile, 'w', encoding='utf-8') as log_file:
            p = OrderedDict()
            p['dataset'] = getattr(self.args, 'dataset', '')
            p['task_name'] = getattr(self.args, 'task_name', '')
            p['backbone'] = getattr(self.args, 'backbone', '')
            p['out_stride'] = getattr(self.args, 'out_stride', '')
            p['lr'] = getattr(self.args, 'lr', '')
            p['lr_scheduler'] = getattr(self.args, 'lr_scheduler', '')
            p['loss_type'] = getattr(self.args, 'loss_type', '')
            p['epochs'] = getattr(self.args, 'epochs', '')
            p['base_size'] = getattr(self.args, 'base_size', '')
            p['crop_size'] = getattr(self.args, 'crop_size', '')
            p['train_resize_mode'] = getattr(self.args, 'train_resize_mode', '')
            p['eval_resize_mode'] = getattr(self.args, 'eval_resize_mode', '')
            p['selected_classes'] = getattr(self.args, 'selected_classes', '')
            p['manifest_dir'] = getattr(self.args, 'manifest_dir', '')
            p['config'] = getattr(self.args, 'config', '')

            for key in sorted(vars(self.args).keys()):
                if key not in p:
                    p[key] = getattr(self.args, key)

            for key, val in p.items():
                log_file.write(key + ':' + str(val) + '\n')
