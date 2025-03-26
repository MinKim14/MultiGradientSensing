import os
import os.path as osp
import wandb


class WandbLogger:
    def __init__(self, log_dir, name, project_name="sensor_ai", test=False):
        super().__init__()
        self.log_dir = log_dir
        self.test = test
        if test:
            return

        wandb.init(project=project_name, name=name)
        wandb.define_metric("custom_step")

        self.wandb_log_keys = {}

        self.name = name

    def wandb_log(self, log_name, value, epoch=None, batch_idx=None):
        if log_name not in self.wandb_log_keys:
            self.wandb_log_keys[log_name] = 0

        wandb.log({log_name: value, "custom_step": self.wandb_log_keys[log_name]})
        self.wandb_log_keys[log_name] += 1


class CsvLogger:
    def __init__(self, log_dir, name):
        super().__init__()
        self.log_dir = log_dir

        self.name = name
        self.train_log_file = osp.join(self.log_dir, self.name + "_train.csv")
        self.val_log_file = osp.join(self.log_dir, self.name + "_val.csv")
        self.test_log_file = osp.join(self.log_dir, self.name + "_test.csv")

        self.train_log = open(self.train_log_file, "w")
        self.val_log = open(self.val_log_file, "w")
        self.test_log = open(self.test_log_file, "w")

    def write_train(self, log):
        self.train_log.write(str(log) + "\n")
        self.train_log.flush()

    def write_val(self, log):
        self.val_log.write(str(log) + "\n")
        self.val_log.flush()

    def write_test(self, log):
        self.test_log.write(str(log) + "\n")
        self.test_log.flush()


class CsvLoggerSingle:
    def __init__(self, log_dir, num_test_cases, name, test_names):
        super().__init__()
        self.log_dir = log_dir

        self.name = name
        self.num_test_cases = num_test_cases
        self.test_names = test_names

        self.log_file = osp.join(self.log_dir, self.name + ".csv")
        self.log_file = open(self.log_file, "w")

        titles = ["epoch", "train", "train_acc"]
        for i in range(num_test_cases):
            titles.append(test_names[i])
        self.log_file.write(",".join(titles) + "\n")
        self.epoch = 0

    def write_train(self, log):
        self.cur_log = [str(self.epoch), str(log)]

    def write_val(self, log):
        self.cur_log.append(str(log))

    def write_test(self, log):
        self.cur_log.append(str(log))
        if len(self.cur_log) == self.num_test_cases + 3:
            self.log_file.write(",".join(self.cur_log) + "\n")
            self.log_file.flush()
            self.epoch += 1
