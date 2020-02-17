from fastai import *
from fastai.text import * 
from fastai.distributed import *
from fastai.distributed import OurDistributedSampler

def _learner_my_distributed(learn:Learner, cuda_id:int, cache_dir:PathOrStr='tmp'):
    "Put `learn` on distributed training with `cuda_id`."
    learn.callbacks.append(MyDistributedTrainer(learn, cuda_id))
    learn.callbacks.append(DistributedRecorder(learn, cuda_id, cache_dir))
    return learn

Learner.to_my_distributed = _learner_my_distributed

class MyDistributedTrainer(LearnerCallback):
    _order = -20 # Needs to run before the recorder
    def __init__(self, learn:Learner, cuda_id:int=0):
        super().__init__(learn)
        self.cuda_id,self.train_sampler = cuda_id,None

    def _change_dl(self, dl, shuffle):
        old_dl = dl
        sampler = OurDistributedSampler(dl.dataset, shuffle=shuffle)
        new_dl = dl.new(shuffle=False, sampler=sampler)
        return old_dl,new_dl,sampler

    def on_train_begin(self, **kwargs):
        self.learn.model = DistributedDataParallel(self.model, device_ids=[self.cuda_id], output_device=self.cuda_id, find_unused_parameters=True)
        shuffle = self.data.train_dl.init_kwargs['shuffle'] if hasattr(self.data.train_dl, 'init_kwargs') else True
        self.old_train_dl,self.data.train_dl,self.train_sampler = self._change_dl(self.data.train_dl, shuffle)
        if hasattr(self.data, 'valid_dl') and self.data.valid_dl is not None:
            self.old_valid_dl,self.data.valid_dl,self.valid_sampler = self._change_dl(self.data.valid_dl, shuffle)
        self.rank = rank_distrib()
        self.recorder.silent = (self.rank != 0)

    def on_epoch_begin(self, epoch, **kwargs): self.train_sampler.set_epoch(epoch)

    def on_train_end(self, **kwargs):
        self.learn.model = self.learn.model.module
        self.learn.data.train_dl = self.old_train_dl
        if hasattr(self.learn.data, 'valid_dl') and self.learn.data.valid_dl is not None:
            self.learn.data.valid_dl = self.old_valid_dl