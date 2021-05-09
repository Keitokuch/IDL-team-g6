import os
import torch.nn as nn

class Session():
    def __init__(self, name, model_factory=None, optim_factory=None, 
                 criterion=nn.CrossEntropyLoss(reduction='none'),
                 train_data=None, val_data=None, sched_factory=None, use_amp=False, **kwargs):
        self.name = name
        self.model_factory = model_factory or self._load_model
        self.optim_factory = optim_factory or self._get_optim
        self.train_data = train_data
        self.val_data = val_data
        self.criterion = criterion
        self.sched_factory = sched_factory
        self.use_amp = use_amp
        self.init_session()
        self.init(**kwargs)
        if os.path.exists(name):
            if self.load_checkpoint("last"):
                print("Restored to epoch", self.epoch)
        elif model_factory is None:
            raise Exception(f"Session {name} doesn't exist and model_factory is not provided!")
        else:
            self.save()

    """ Overridables """
    def init(self, **kwargs):
        pass

    def run_epoch(epoch, end_epoch):
        pass

    def train_model(self):
        pass

    def eval_model(self):
        pass

    def get_checkpoint(self):
        return {}

    def put_checkpoint(self, ckpt_dict):
        pass
    """"""
    
    def train(self, num_epochs=60, checkpoint=-1, lr=None, disable_sched=False):
        if type(checkpoint) is int:
            if checkpoint > 0:
                self.load_checkpoint(checkpoint)
            elif checkpoint == 0:
                self.init_session()
        else:
            self.load_checkpoint(checkpoint)
        if lr:
            self.set_lr(lr)
        use_sched = True if self.lr_sched is not None and not disable_sched else False
        end_epoch = self.epoch + num_epochs
        for i in range(num_epochs):
            self.epoch += 1
            score, loss = self.run_epoch(self.epoch, end_epoch)

            if use_sched:
                self.lr_sched.step(self.epoch, score, loss)
            self.save_checkpoint()

    def save(self, name=None):
        name = name or self.name
        try:
            os.makedirs(name)
        except: 
            pass
        torch.save(self.model, os.path.join(self.name, 'model_proto.torch'))

    def init_session(self):
        self.epoch = 0
        self.model = self.model_factory().to(device)
        self.optim = self.optim_factory(self.model)
        if self.sched_factory:
            self.lr_sched = self.sched_factory(self.optim)
        else:
            self.lr_sched = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _load_model(self):
        return torch.load(os.path.join(self.name, 'model_proto.torch'))

    def _get_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=1e-3)

    def save_checkpoint(self, cp_name=None, sess_name=None):
        name = sess_name or self.name
        cp_name = str(cp_name) if cp_name is not None else str(self.epoch)
        ckpt_path = os.path.join(name, cp_name)
        ckpt = {
            'model_state': self.model.state_dict(),
            'optim_state': self.optim.state_dict(),
            'epoch': self.epoch
        }
        if self.use_amp:
            ckpt['scaler'] = self.scaler.state_dict()
        ckpt.update(self.get_checkpoint())
        torch.save(ckpt, ckpt_path)
        last = os.path.join(name, "last")
        torch.save(ckpt, last)

    def load_checkpoint(self, cp_name):
        cp_name = str(cp_name)
        ckpt_path = os.path.join(self.name, cp_name)
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} doesn't exist.")
            return None
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model_state'])
        self.optim.load_state_dict(ckpt['optim_state'])
        if 'scaler' in ckpt and ckpt['scaler']:
            self.scaler.load_state_dict(ckpt['scaler'])
        if 'epoch' in ckpt:
            self.epoch = ckpt['epoch']
        self.put_checkpoint(ckpt)
        print("Loaded checkpoint", ckpt_path)
        return self

    def set_optim(self, optim_factory):
        self.optim_factory = optim_factory
        self.optim = optim_factory(model)
        if self.lr_sched:
            self.lr_sched = self.sched_factory(self)
        return self

    def set_lr_sched(self, sched_factory):
        self.sched_factory = sched_factory
        self.lr_sched = sched_factory(self.optim)
        return self

    def set_lr(self, lr):
        for group in self.optim.param_groups:
            group['lr'] = lr
        print('Setting lr =', lr)
        if self.lr_sched:
            self.lr_sched.lr = lr
        return self

    def save_as(self, name):
        from shutil import copytree
        copytree(self.name, name)
        self.name = name
