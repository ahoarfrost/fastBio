#data loading and bunching; defines the dataloader and items
#contains the basic item Sequence, the basic itemlist SeqList, 

from Bio import SeqIO
from fastai import *
from fastai.basics import *
from fastai.data_block import *
from fastai.text import LanguageModelPreLoader #TODO: remove fastai.text dependencies
from iterators import *
from transform import *


seqfiletype_to_iterator = {
    '.fastq': FastqIterator
}

def check_seqfiletype(filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes):
    if isinstance(filename, Path): 
        seqfiletype = filename.suffix
    else:
        seqfiletype = '.'+filename.split('.')[-1]
    assert seqfiletype in extensions, "Input sequence file type %r is not supported." % seqfiletype
    return seqfiletype

def open_single_read(filename, offset, tok, vocab, sep, extensions:Collection[str]=supported_seqfiletypes):
    #opens a single sequence from a particular offset in a seqfile, tokenizes, numericalizes, and returns as Sequence object
    seqfiletype = check_seqfiletype(filename, extensions)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename,"r") as handle:
        handle.seek(offset)
        title, seq, qual, off = next(iterator(handle, offset))
    handle.close()
    #our seq is a string. tokenize & numericalize our seq 
    tokens = tok.tokenizer(seq)
    ids = vocab.numericalize(tokens)
    seqout = vocab.textify(ids, sep=sep)
        
    #and return as Sequence
    return Sequence(ids, seqout)

def get_items_from_seqfile(filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes):
    seqfiletype = check_seqfiletype(filename, extensions)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename, "r") as handle:
        items = []
        for title, seq, qual, offset in iterator(handle):
            items.append((Path(filename), offset))
    handle.close()
    return items


class BioDataBunch(DataBunch):
    "General class to get a `DataBunch` for bio sequences. Subclassed by `BioClasDataBunch` and `BioLMDataBunch`."
    #todo: from_folder, from_seqfile from_csv (a file with labels and filenames)
    #creates seqlist, splits, labels, (optionally) add test, add transforms to be applied, and finally converts to databunch
    
    @classmethod
    def from_seqfile(cls, path:PathOrStr, filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, 
                        valid_pct:float=0.2, seed:int=None, label_func=None, 
                        vocab:BioVocab=BioVocab.create_from_ksize(), tokenizer:BioTokenizer=BioTokenizer(), pad_idx:int=1, sep=' ', **kwargs:Any) -> 'BioDataBunch':
        "Create from a sequence file."
        seqlist = SeqList.from_seqfile(filename=filename, path=path, extensions=extensions)
        src = seqlist.split_by_rand_pct(valid_pct=valid_pct, seed=seed)
        if cls==BioLMDataBunch:
            src = src.label_for_lm()
        else:
            src = src.label_from_func(label_func)

        return src.databunch(**kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr='.', train:PathOrStr='train',valid:PathOrStr='valid', test:Optional[PathOrStr]=None, valid_pct:float=None, 
                    extensions:Collection[str]=supported_seqfiletypes, recurse:bool=True,
                    vocab:BioVocab=BioVocab.create_from_ksize(), tokenizer:BioTokenizer=BioTokenizer(), pad_idx:int=1, sep=' ', 
                    seed:int=None, classes:Collection=None, **kwargs:Any) -> 'BioDataBunch':
        "Create from seqfiles in a folder (or nested folders). These seqfiles should either be sorted into 'train', 'valid', and 'test' subfolders, or provide a 'valid_pct' to randomly split a percent of datasets."
        seqlist = SeqList.from_folder(path=path, extensions=extensions, recurse=recurse)
        if valid_pct is None: 
            src = seqlist.split_by_folder(train=train, valid=valid)
        else:
            src = seqlist.split_by_rand_pct(valid_pct, seed)
        if cls==BioLMDataBunch:
            src = src.label_for_lm()
        else:
            src = src.label_from_folder(classes=classes)
        if test is not None:
            src.add_test_folder(path/test)
        
        return src.databunch(**kwargs)

class BioLMDataBunch(BioDataBunch): 
    "Create a BioDataBunch for training a language model."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', no_check:bool=False, bs:int=64, val_bs:int=None,
               num_workers:int=0, device:torch.device=None, collate_fn:Callable=data_collate,
               dl_tfms:Optional[Collection[Callable]]=None, bptt:int=70, backwards:bool=False, **dl_kwargs) -> DataBunch:
        "Create a `BioDataBunch` in `path` from the `datasets` for language modelling. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards)
                    for i,ds in enumerate(datasets)]
        val_bs = bs
        dls = [DataLoader(d, b, shuffle=False, **dl_kwargs) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]

        return cls(*dls, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)

class Sequence(ItemBase):
    "Basic item for biological sequence data."
    #ids are an array of numbers, of numericalized vocab IDs for that sequence; 
    # seq is the sequence of tokens for the sequence, separated by the 'sep' (default is a space) (output of BioVocab.textify)
    def __init__(self, ids, seq): 
        self.data = np.array(ids, dtype=np.int64)
        self.seq = seq
        
    def __str__(self): 
        return str(self.seq)
        
class SeqList(ItemList):
    "Basic `ItemList` for biological sequence data."
    _bunch = BioLMDataBunch #note this should probably be BioClasDataBunch, but still need to write it
    #_processor = [BioTokenizeProcessor, BioNumericalizeProcessor]
    _is_lm = False

    def __init__(self, items:Iterator, vocab:BioVocab=BioVocab.create_from_ksize(), tokenizer:BioTokenizer=BioTokenizer(), pad_idx:int=1, sep=' ', extensions:Collection[str]=supported_seqfiletypes, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.tokenizer,self.pad_idx,self.sep, self.extensions = vocab,tokenizer,pad_idx,sep,extensions
        self.copy_new += ['vocab', 'tokenizer', 'pad_idx', 'sep', 'extensions']

    def get(self, i):
        #o is the ith index of self.items. This is a tuple of a filename and an offset value of the read's location in the file
        filename,offset = super().get(i)
        #use an iterator to go to the file at the right line, get the record info, tokenize/numericalize as needed, and return a Sequence() object
        one_sequence = open_single_read(filename=filename, offset=offset, tok=self.tokenizer, vocab=self.vocab, sep=self.sep, extensions=self.extensions)
 
        return one_sequence

    def label_for_lm(self, **kwargs):
        "A special labelling method for language models."
        self.__class__ = LMSeqList
        kwargs['label_cls'] = LMLabelList
        return self.label_const(0, **kwargs)

    def reconstruct(self, t:Tensor):
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Sequence(t[idx_min:idx_max+1], self.vocab.textify(t[idx_min:idx_max+1], sep=self.sep))

    @classmethod
    def from_seqfile(cls, filename:PathOrStr, path:PathOrStr='.', extensions:Collection[str]=supported_seqfiletypes, **kwargs)->'SeqList':
        "Creates a SeqList from a single sequence file (e.g. .fastq, .fasta, etc.)"
        #get (filename, offset) tuple for each read and add to items
        items = get_items_from_seqfile(filename=filename, extensions=extensions)
        
        return cls(items=items, path=path, **kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=supported_seqfiletypes, recurse:bool=True, **kwargs) -> 'SeqList':
        "Creates a SeqList from all sequence files in a folder"
        #get list of files in `path` with seqfile suffixes. `recurse` determines if we search subfolders.
        files = get_files(path=path, extensions=extensions, recurse=recurse)
        #within each file, get (filename, offset) tuple for each read and add to items
        items = []
        for filename in files:
            items.extend(get_items_from_seqfile(filename=filename, extensions=extensions))

        print(items)
        return cls(items=items, path=path, **kwargs)


    def show_xys(self, xs, ys, max_len:int=70)->None:
        "Show the `xs` (inputs) and `ys` (targets). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        names = ['idx','seq'] if self._is_lm else ['seq','target']
        items = []
        for i, (x,y) in enumerate(zip(xs,ys)):
            txt_x = ' '.join(x.seq.split(' ')[:max_len]) if max_len is not None else x.seq
            items.append([i, txt_x] if self._is_lm else [txt_x, y])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))

    def show_xyzs(self, xs, ys, zs, max_len:int=70):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions). `max_len` is the maximum number of tokens displayed."
        from IPython.display import display, HTML
        items,names = [],['seq','target','prediction']
        for i, (x,y,z) in enumerate(zip(xs,ys,zs)):
            txt_x = ' '.join(x.seq.split(' ')[:max_len]) if max_len is not None else x.seq
            items.append([txt_x, y, z])
        items = np.array(items)
        df = pd.DataFrame({n:items[:,i] for i,n in enumerate(names)}, columns=names)
        with pd.option_context('display.max_colwidth', -1):
            display(HTML(df.to_html(index=False)))


class LMLabelList(EmptyLabelList):
    "Basic `ItemList` for dummy labels."
    def __init__(self, items:Iterator, **kwargs):
        super().__init__(items, **kwargs)
        self.loss_func = CrossEntropyFlat()

class LMSeqList(SeqList):
    "Special `SeqList` for a language model."
    _bunch = BioLMDataBunch
    _is_lm = True