import glob
import os

import json
import torch


class CheckpointSaver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkpoint)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        self.save_experiment_config()

    def save_checkpoint(self, state, filename='model.pth.tar', write_best=True):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)

    def save_experiment_config(self):
        with open(os.path.join(self.experiment_dir, 'parameters.txt'), 'w') as file:
            config_dict = vars(self.args)
            for k in vars(self.args):
                file.write(f"{k}={config_dict[k]} \n")

    def save_experiment_epoch_log(self, log: str, verbose: bool = False):
        if verbose:
            print("Save log to: ", os.path.join(self.experiment_dir, 'result_epoch_log.txt'))
        with open(os.path.join(self.experiment_dir, 'result_epoch_log.txt'), 'a') as file:
            file.write(log + "\n")

    def save_testing_results_json(self, results: dict, name: str):
        json_path = os.path.join(self.experiment_dir, name + '.json')
        with open(json_path, 'wt') as fp:
            json.dump(results, fp, indent=4)
        print("[CheckpointSaver] :: Save to Json file !  dict name = ", name, ", save to: ", json_path)

    def load_testing_results_json(self, name: str):
        json_path = os.path.join(self.experiment_dir, name + '.json')
        with open(json_path, 'rt') as fp:
            jfile = json.load(fp)
        print("[CheckpointSaver] :: Load Json file !  dict name = ", name, ", save to: ", json_path)
        return jfile
