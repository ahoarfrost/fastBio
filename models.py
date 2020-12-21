from fastai import *
from fastai.text import *
from data import *
import urllib.request

class LookingGlass(LanguageLearner):
    "loads the LookingGlass model, with pretrained LookingGlass weights if pretrained=True (true by default)"
    def __init__(self, data:DataBunch):
        config = awd_lstm_lm_config.copy()
        config['n_layers'] = 3
        config['n_hid'] = 1152
        config['bidir'] = False
        config['emb_sz'] = 104
        self.config = config
        self.drop_mult = 0.1
        self.data = data

    def load(self, pretrained:bool=True, pretrained_dir=None):
        model = get_language_model(AWD_LSTM, len(self.data.vocab.itos), config=self.config, drop_mult=self.drop_mult)
        learn = LanguageLearner(self.data, model, split_func=awd_lstm_lm_split)
        learn.data.bptt = 100

        if pretrained:
            if pretrained_dir:
                model_path = Path(pretrained_dir)
            else:
                model_path = Path('./').resolve()

            if not Path(model_path/'LookingGlass.pth').exists():
                print('downloading pretrained model to',str(Path(model_path/'LookingGlass.pth')))
                model_url = 'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass.pth'
                urllib.request.urlretrieve(model_url, "LookingGlass.pth")
            
            print('loading pretrained LookingGlass language model')
            learn = learn.load(str(Path(model_path/'LookingGlass')))
            learn.freeze()

        return learn
