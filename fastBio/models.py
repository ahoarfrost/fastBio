from fastai import *
from fastai.text import *
from data import *
import urllib.request

pretrained_urls = {
    'LookingGlass':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass.pth',
    'LookingGlass_enc':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/LookingGlass_enc.pth',
    'FunctionalClassifier_enc':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/FunctionalClassifier_enc.pth',
    'FunctionalClassifier':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/FunctionalClassifier.pth',
    'OptimalTempClassifier':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/OptimalTempClassifier.pth',
    'OxidoreductaseClassifier':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/OxidoreductaseClassifier.pth',
    'ReadingFrameClassifier':'https://github.com/ahoarfrost/LookingGlass/releases/download/v1.0/ReadingFrameClassifier.pth'
}

class LookingGlass():
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
                model_url = pretrained_urls['LookingGlass']
                urllib.request.urlretrieve(model_url, "LookingGlass.pth")
            
            print('loading pretrained LookingGlass language model')
            learn.load(str(Path(model_path/'LookingGlass')))
            learn.freeze()

        return learn

class LookingGlassClassifier():
    "load a classifier model with 1) the pretrained LookingGlass encoder architecture or 2) a pretrained fine tuned LG classifier (by name) as described in the associated paper"
    def __init__(self, data:DataBunch):
        config = awd_lstm_clas_config.copy() #make sure this is awd_lstm_clas_config for classification!
        config['bidir'] = False #this can be True for classification, but if you're using a pretrained LM you need to keep if = False because otherwise your matrix sizes don't match
        config['n_layers'] = 3
        config['n_hid'] = 1152
        config['emb_sz'] = 104
        self.config = config
        self.drop_mult = 0.3
        self.data = data

    def load_encoder(self, pretrained_name:str='LookingGlass_enc', pretrained:bool=True, pretrained_dir=None):
        model = get_text_classifier(AWD_LSTM, len(self.data.vocab.itos), self.data.c, 
                                config=self.config, drop_mult=self.drop_mult)

        learn = RNNLearner(self.data, model, split_func=awd_lstm_clas_split)
        
        if pretrained:
            if pretrained_dir:
                model_path = Path(pretrained_dir)
            else:
                model_path = Path('./').resolve()

            if not Path(model_path/Path(pretrained_name+'.pth')).exists():
                print('downloading pretrained model to',str(Path(model_path/Path(pretrained_name+'.pth'))))
                model_url = pretrained_urls[pretrained_name]
                urllib.request.urlretrieve(model_url, pretrained_name+".pth")
            
            print('loading classifier with pretrained encoder from',str(Path(model_path/Path(pretrained_name+'.pth'))))
            learn.load_encoder(str(Path(model_path/Path(pretrained_name))))
            learn.freeze()

        return learn

    def load(self, pretrained_name:str='FunctionalClassifier', pretrained:bool=True, pretrained_dir=None):
        model = get_text_classifier(AWD_LSTM, len(self.data.vocab.itos), self.data.c, 
                                config=self.config, drop_mult=self.drop_mult)

        learn = RNNLearner(self.data, model, split_func=awd_lstm_clas_split)
        
        if pretrained:
            if pretrained_dir:
                model_path = Path(pretrained_dir)
            else:
                #save to working directory if no directory provided
                model_path = Path('./').resolve()

            if not Path(model_path/Path(pretrained_name+'.pth')).exists():
                print('downloading pretrained model to',str(Path(model_path/Path(pretrained_name+'.pth'))))
                model_url = pretrained_urls[pretrained_name]
                urllib.request.urlretrieve(model_url, pretrained_name+".pth")
            
            print('loading pretrained classifier',pretrained_name)
            learn.load(str(Path(model_path/Path(pretrained_name))))
            learn.freeze()

        return learn