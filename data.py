#data loading and bunching; defines the dataloader and items
#contains the basic item Sequence, the basic itemlist SeqList, 

from Bio import SeqIO
from fastai import *
from fastai.basics import *
from fastai.data_block import *
from iterators import *
from transform import *

supported_seqfiletypes = ['.fastq']

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

def open_single_read(filename, offset, tok, vocab, extensions:Collection[str]=supported_seqfiletypes):
    seqfiletype = check_seqfiletype(filename, extensions)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename,"r") as handle:
        handle.seek(offset)
        title, seq, qual, off = next(iterator(handle, offset))
    #our seq is a string. tokenize & numericalize our seq 
    tokens = tok.tokenizer(seq)
    ids = vocab.numericalize(tokens)
        
    #and return as Sequence
    return Sequence(ids, tokens)

def get_items_from_seqfile(filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes):
    seqfiletype = check_seqfiletype(filename, extensions)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename, "r") as handle:
        items = []
        for title, seq, qual, offset in iterator(handle):
            items.append((Path(filename), offset))
    return items


class Sequence(ItemBase):
    "Basic item for biological sequence data."
    #ids are an array of numbers, of numericalized vocab IDs for that sequence; 
    # seq is the sequence of tokens for the sequence, separated by the 'sep' (default is a space) (output of BioVocab.textify)
    def __init__(self, ids, tokens): 
        self.data = np.array(ids, dtype=np.int64)
        self.seq = tokens
        
    def __str__(self): 
        return str(self.seq)

class SeqList(ItemList):
    "Basic `ItemList` for biological sequence data."
    #_bunch = BioClasDataBunch
    #_processor = [BioTokenizeProcessor, BioNumericalizeProcessor]
    _is_lm = False

    def __init__(self, items:Iterator, vocab=BioVocab.create(), tokenizer=BioTokenizer(), pad_idx:int=1, sep=' ', extensions:Collection[str]=supported_seqfiletypes, **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.tokenizer,self.pad_idx,self.sep, self.extensions = vocab,tokenizer,pad_idx,sep,extensions
        self.copy_new += ['vocab', 'tokenizer', 'pad_idx', 'sep', 'extensions']

    def get(self, i):
        #o is the ith index of self.items. This is a tuple of a filename and an offset value of the read's location in the file
        filename,offset = super().get(i)
        #use a FastqIterator to go to the file at the right line, get the record info, tokenize/numericalize as needed, and return a Sequence() object
        one_sequence = open_single_read(filename=filename, offset=offset, tok=self.tokenizer, vocab=self.vocab, extensions=self.extensions)
 
        return one_sequence

    def label_for_lm(self, **kwargs):
        "A special labelling method for language models."
        self.__class__ = LMSeqList
        kwargs['label_cls'] = LMLabelList
        return self.label_const(0, **kwargs)

    def reconstruct(self, t:Tensor):
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Sequence(t[idx_min:idx_max+1], self.vocab.textify(t[idx_min:idx_max+1]))

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
        files = get_files(path=path, extensions=extensions, recurse=recurse, **kwargs)
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
    #_bunch = BioLMDataBunch
    _is_lm = True

'''
class BioDataBunch(DataBunch):
    "General class to get a `DataBunch` for bio sequences. Subclassed by `BioClasDataBunch` and `BioLMDataBunch`."
    #todo: from_folder, from_seqfile from_csv (a file with labels and filenames)
    #creates seqlist, splits, labels, (optionally) add test, add transforms to be applied, and finally converts to databunch
    
    def create_from_ll(cls, lls:LabelLists, bs:int=64, val_bs:int=None, ds_tfms:Optional[TfmList]=None, 
                num_workers:int=defaults.cpus, dl_tfms:Optional[Collection[Callable]]=None, device:torch.device=None,
                test:Optional[PathOrStr]=None, collate_fn:Callable=data_collate, size:int=None, no_check:bool=False,
                resize_method:ResizeMethod=None, mult:int=None, padding_mode:str='reflection',
                mode:str='bilinear', tfm_y:bool=False)->'BioDataBunch':
    )
    #@classmethod
    #def from_seqfile()

    @classmethod
    def from_folder ()
    
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = _get_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields, 
                                   include_bos=include_bos, include_eos=include_eos)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        if cls==TextLMDataBunch: src = src.label_for_lm()
        else: 
            if label_delim is not None: src = src.label_from_df(cols=label_cols, classes=classes, label_delim=label_delim)
            else: src = src.label_from_df(cols=label_cols, classes=classes)
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)

    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, chunksize:int=10000, max_vocab:int=60000,
                min_freq:int=2, mark_fields:bool=False, bptt=70, collate_fn:Callable=data_collate, bs=64, **kwargs):
        "Create a `TextDataBunch` from DataFrames. `kwargs` are passed to the dataloader creation."
        processor = _get_genomic_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, mark_fields=mark_fields)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),
                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))
        src = src.label_for_lm() 
        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        d1 = src.databunch(**kwargs)
        
        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)            
        val_bs = bs
        datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=False) 
                    for i,ds in enumerate(datasets)]            
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        
        return cls(*dls, path=path, collate_fn=collate_fn, no_check=False)
'''