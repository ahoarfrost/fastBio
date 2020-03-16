from fastai import *
from fastai.text import *
from fastai.callbacks import *

def get_encpreds(model:nn.Module, dl:DataLoader, pbar:Optional[PBar]=None, cb_handler:Optional[CallbackHandler]=None,
              activ:nn.Module=None, loss_func:OptLossFunc=None, n_batch:Optional[int]=None) -> List[Tensor]:
    "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
    res = get_layer_out(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch)
    #res = [to_float(o.cpu()) for o in
    #        get_layer_out(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch)]
    #       zip(*validate(model, dl, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
    if loss_func is not None:
        with NoneReduceOnCPU(loss_func) as lf: res.append(lf(res[0], res[1]))
    #if activ is not None: res[0] = activ(res[0])
    return res

def get_layer_out(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average:bool=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        embs,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            if not is_listy(xb): xb = [xb]
            out = model(*xb)
            raw_out = out[0]
            last_layer_out = raw_out[-1] #list of length batch_size, each item is seqlen x emb_sz array of each token's model output
            last_token_out = last_layer_out[:,-1] #tensor of length batch_size, each item is tensor of length emb_sz
            emblist = last_token_out.tolist()
            embs.extend(np.array(emblist))
            #embs.append(last_token_out)
            
            if not is_listy(yb): yb = yb.tolist()
            nums.extend(yb)
            #nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(embs[-1]): break
            if n_batch and (len(nums)>=n_batch): break

        #nums = np.array(nums, dtype=np.float32)
        '''
        if average: 
            #could take max/min/avg concat of embs here maybe?
            return (to_np(torch.stack(embs)) * nums).sum() / nums.sum()
        else:       return val_losses
        '''
        return embs,nums

class RNNEncLearner(RNNLearner):
    "Basic class for a `Learner` in NLP."
    def __init__(self, data:DataBunch, model:nn.Module, split_func:OptSplitFunc=None, clip:float=None,
                 alpha:float=2., beta:float=1., metrics=None, **learn_kwargs):
        is_class = (hasattr(data.train_ds, 'y') and (isinstance(data.train_ds.y, CategoryList) or
                                                     isinstance(data.train_ds.y, LMLabelList)))
        metrics = ifnone(metrics, ([accuracy] if is_class else []))
        super().__init__(data,model,split_func,clip,alpha,beta,metrics,**learn_kwargs)
        self.callbacks.append(RNNTrainer(self, alpha=alpha, beta=beta))
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if split_func: self.split(split_func)

    def get_preds(self, ds_type:DatasetType=DatasetType.Valid, activ:nn.Module=None, with_loss:bool=False, n_batch:Optional[int]=None,
                  pbar:Optional[PBar]=None, ordered:bool=True) -> List[Tensor]:
        "Return predictions and targets on the valid, train, or test set, depending on `ds_type`."
        self.model.reset()
        #preds = super().get_preds(ds_type=ds_type, activ=activ, with_loss=with_loss, n_batch=n_batch, pbar=pbar)
        lf = self.loss_func if with_loss else None
        #activ = ifnone(activ, _loss_func2activ(self.loss_func))
        if not getattr(self, 'opt', False): self.create_opt(defaults.lr, self.wd)
        callbacks = [cb(self) for cb in self.callback_fns + listify(defaults.extra_callback_fns)] + listify(self.callbacks)
        if ordered: np.random.seed(42)
        preds = get_encpreds(self.model, self.dl(ds_type), cb_handler=CallbackHandler(callbacks),
                         activ=activ, loss_func=lf, n_batch=n_batch, pbar=pbar)

        if ordered and hasattr(self.dl(ds_type), 'sampler'):
            np.random.seed(42)
            sampler = [i for i in self.dl(ds_type).sampler]
            reverse_sampler = np.argsort(sampler)
            predx = [preds[0][x] for x in reverse_sampler]
            predy = [preds[1][x] for x in reverse_sampler]
            preds = (predx,predy)
            #preds = [p[reverse_sampler] for p in preds]
        return preds
