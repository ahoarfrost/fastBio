#data loading and bunching

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from fastai import *
from iterator import *

supported_seqfiletypes = ['fastq']

seqfiletype_to_iterator = {
    'fastq': FastqIterator
}

def check_seqfiletype(seqfiletype):
    assert seqfiletype in supported_seqfiletypes, "Input sequence file type %r is not supported." % seqfiletype

def open_single_read(filename, offset, seqfiletype, tok, vocab):
    check_seqfiletype(seqfiletype)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename,"r") as handle:
        handle.seek(offset)
        title, seq, qual, off = next(iterator(handle, offset))
    #our seq is a string. tokenize our seq 
    tokens = tok.tokenizer(seq)
    ids = vocab.numericalize(tokens)
        
    #and return as Sequence
    return Sequence(ids, tokens)


class Sequence(ItemBase):
    "Basic item for biological sequence data."
    #ids are the array of numericalized vocab IDs for that sequence; 
    # seq is the sequence of tokens for the sequence, separated by the 'sep' (default is a space) (output of BioVocab.textify)
    def __init__(self, ids, tokens): 
        self.data = np.array(ids, dtype=np.int64)
        self.seq = tokens
        
    def __str__(self): 
        return str(self.seq)

class SeqList(ItemList):
    "Basic `ItemList` for biological sequence data."
    _bunch = BioClasDataBunch
    _processor = [BioTokenizeProcessor, BioNumericalizeProcessor]
    _is_lm = False

    def __init__(self, items:Iterator, vocab:BioVocab=None, tokenizer:BioTokenizer=None, pad_idx:int=0, sep=' ', seqfiletype='fastq', **kwargs):
        super().__init__(items, **kwargs)
        self.vocab,self.tokenizer,self.pad_idx,self.sep, self.seqfiletype = vocab,tokenizer,pad_idx,sep,seqfiletype
        self.copy_new += ['vocab', 'tokenizer', 'pad_idx', 'sep', 'seqfiletype']

    def get(self, i):
        #o is the ith index of self.items. This is a tuple of a filename and an offset value of the read's location in the file
        filename,offset = super().get(i)
        #use a FastqIterator to go to the file at the right line, get the record info, tokenize/numericalize as needed, and return a Sequence() object
        one_sequence = open_single_read(filename, offset, self.seqfiletype, self.tokenizer, self.vocab)
 
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
    def from_seqfile(cls, path:PathOrStr='.', extensions:Collection[str]=seq_extensions, vocab:BioVocab=None,
                    processor:PreProcessor=None, **kwargs)->'SeqList':
        "Creates a SeqList from a single sequence file (e.g. .fastq, .fasta, etc.)"
        #get (filename, offset) tuple for each read and add to items
        check_seqfiletype(self.seqfiletype)
        iterator = seqfiletype_to_iterator[self.seqfiletype]
        with open(filename, "r") as handle:
            items = []
            for title, seq, qual, offset in iterator(handle):
                items.append((filename, offset))
        
        #processor = ifnone(processor, [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        
        return cls(items=items, path=path, processor=processor, **kwargs)


    @classmethod
    #def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]):
    #    "Creates a SeqList from all sequence files in a folder"
        #get list of files in `path` with seqfile suffixes. `recurse` determines if we search subfolders.
        #within each file, get (filename, offset) tuple for each read and add to items

    @classmethod #compare to TextList
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=text_extensions, vocab:Vocab=None,
                    processor:PreProcessor=None, **kwargs)->'TextList':
        "Get the list of files in `path` that have a text suffix. `recurse` determines if we search subfolders."
        processor = ifnone(processor, [OpenFileProcessor(), TokenizeProcessor(), NumericalizeProcessor(vocab=vocab)])
        return super().from_folder(path=path, extensions=extensions, processor=processor, **kwargs)



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

class LMTextList(SeqList):
    "Special `SeqList` for a language model."
    _bunch = TextLMDataBunch
    _is_lm = True

class BioDataBunch(DataBunch):
    "General class to get a `DataBunch` for bio sequences. Subclassed by `BioClasDataBunch` and `BioLMDataBunch`."
    #todo: from_fastas (a folder with whole fasta files with multiple sequences e.g. genomes or metagenomes); from_csv (a file with file names and ); from_folder
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