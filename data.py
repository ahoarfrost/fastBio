#data loading and bunching; defines the dataloader and items
#contains the basic item Sequence, the basic itemlist SeqList, 

from fastai import *
from fastai.text import *
from iterators import *
from transform import *

supported_seqfiletypes = ['.fastq', '.fastq.abundtrim', '.fna', '.fasta', '.ffn', '.faa']

seqfiletype_to_iterator = {
    '.fastq': FastqIterator,
    '.fastq.abundtrim': FastqIterator,
    '.fna': FastaIterator,
    '.fasta': FastaIterator,
    '.ffn': FastaIterator,
    '.faa': FastaIterator
}

def _get_bio_files(parent, p, f, extensions):
    p = Path(p)#.relative_to(parent)
    if isinstance(extensions,str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or f'.{".".join(o.split(".")[1:])}' in low_extensions)]
    return res

def get_bio_files(path:PathOrStr, extensions:Collection[str]=None, recurse:bool=True, exclude:Optional[Collection[str]]=None,
              include:Optional[Collection[str]]=None, presort:bool=False, followlinks:bool=False)->FilePathList:
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i==0:   d[:] = [o for o in d if o in include]
            elif exclude is not None and i==0: d[:] = [o for o in d if o not in exclude]
            else:                              d[:] = [o for o in d if not o.startswith('.')]
            res += _get_bio_files(path, p, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_bio_files(path, path, f, extensions)
        if presort: res = sorted(res, key=lambda p: _path_to_same_str(p), reverse=False)
        return res

def check_seqfiletype(filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes):
    if isinstance(filename, Path): 
        seqfiletype = ''.join(filename.suffixes)
    else:
        seqfiletype = '.'+'.'.join(filename.split('.')[1:])
    assert seqfiletype in extensions, "Input sequence file type %r is not supported." % seqfiletype
    return seqfiletype

def get_items_from_seqfile(filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, max_seqs:int=None):
    seqfiletype = check_seqfiletype(filename, extensions)
    iterator = seqfiletype_to_iterator[seqfiletype]
    with open(filename, "r") as handle:
        items = []
        row = 0
        for title, seq, qual, offset in iterator(handle):
            #get (filename, offset, length of sequence) for each read and add to items
            items.append(seq)
            row += 1
            if max_seqs and row >= max_seqs: 
                break
    handle.close()
    return items

class OpenSeqFileProcessor(PreProcessor):
    "`PreProcessor` that opens the filenames and read the sequences. This is used if creating biotextlists from_folder, because need to know the path of each input for splitting."
    def __init__(self, ds:ItemList=None, extensions:Collection[str]=supported_seqfiletypes, max_seqs:int=None):
        self.extensions = extensions
        self.max_seqs = max_seqs

    def process(self, ds:Collection): 
        readitems = []
        for item in ds.items:
            readitems.extend(self.process_one(item, self.extensions, self.max_seqs))
        ds.items = readitems

    def process_one(self,item, extensions, max_seqs): 
        return get_items_from_seqfile(item, extensions=extensions, max_seqs=max_seqs) if isinstance(item, Path) else item


class BioTextList(TextList):
    def __init__(self, items:Iterator, vocab:BioVocab=None, pad_idx:int=1, sep=' ', **kwargs):
        super().__init__(items, **kwargs)
       
    @classmethod
    def from_seqfile(cls, filename:PathOrStr, path:PathOrStr='.', extensions:Collection[str]=supported_seqfiletypes, max_seqs_per_file:int=None, **kwargs)->'TextList':
        "Creates a SeqList from a single sequence file (e.g. .fastq, .fasta, etc.)"
        #get (filename, offset) tuple for each read and add to items
        items = get_items_from_seqfile(filename=filename, extensions=extensions, max_seqs=max_seqs_per_file)
        
        return cls(items=items, path=path, **kwargs)

    @classmethod
    def from_folder(cls, path:PathOrStr='.', vocab:Vocab=None, extensions:Collection[str]=supported_seqfiletypes, 
                        max_seqs_per_file:int=None, recurse:bool=True, processor:PreProcessor=None, **kwargs) -> 'TextList':
        "Creates a SeqList from all sequence files in a folder"
        #get list of files in `path` with seqfile suffixes. `recurse` determines if we search subfolders.
        files = get_bio_files(path=path, extensions=extensions, recurse=recurse)
        #define processor with OpenSeqFileProcessor since items are now a list of filepaths rather than Seq objects
        processor = ifnone(processor, [OpenSeqFileProcessor(), BioTokenizeProcessor(), BioNumericalizeProcessor(vocab=vocab)])
        
        return cls(items=files, path=path, processor=processor, **kwargs)

class BioLMDataBunch(TextLMDataBunch):
    @classmethod
    def from_seqfile(cls, path:PathOrStr, filename:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, max_seqs_per_file:int=None,
                        valid_pct:float=0.2, 
                        bptt=70, bs=64, seed:int=None, collate_fn:Callable=data_collate, device:torch.device=None, no_check:bool=False, backwards:bool=False,
                        vocab:BioVocab=None, tokenizer:BioTokenizer=None, 
                        chunksize:int=10000, max_vocab:int=60000, min_freq:int=2, include_bos:bool=None, include_eos:bool=None,
                        **kwargs:Any):
        "Create from a sequence file."
        processor = get_lol_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, include_bos=include_bos, include_eos=include_eos)

        src = BioTextList.from_seqfile(filename=filename, path=path, extensions=extensions, max_seqs_per_file=max_seqs_per_file, processor=processor)
        src = src.split_by_rand_pct(valid_pct=valid_pct, seed=seed)
        src = src.label_for_lm()

        if test is not None: src.add_test_folder(path/test)

        d1 = src.databunch(**kwargs)

        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)            
        val_bs = bs
        datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards) 
                    for i,ds in enumerate(datasets)]            
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        
        return cls(*dls, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, recurse:bool=True, max_seqs_per_file:int=None,
                    valid_pct:float=0.2, train:str=None, valid:str=None, test:Optional[str]=None,
                    tokenizer:BioTokenizer=None, vocab:BioVocab=None, collate_fn:Callable=data_collate, device:torch.device=None, no_check:bool=False, backwards:bool=False,
                    chunksize:int=10000, max_vocab:int=60000, min_freq:int=2, include_bos:bool=None, include_eos:bool=None, 
                    bptt=70, bs=64, seed:int=None, **kwargs):

        path = Path(path).absolute()

        processor = [OpenSeqFileProcessor(extensions=extensions, max_seqs=max_seqs_per_file)] + get_lol_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, include_bos=include_bos, include_eos=include_eos)

        src = BioTextList.from_folder(path=path, vocab=vocab, extensions=extensions, max_seqs_per_file=max_seqs_per_file, recurse=recurse, processor=processor)                                   
        
        if train and valid:
            src = src.split_by_folder(train=train, valid=valid)
        else:
            src = src.split_by_rand_pct(valid_pct=valid_pct, seed=seed)

        src = src.label_for_lm()

        if test is not None: src.add_test_folder(path/test)

        d1 = src.databunch(**kwargs)

        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)            
        val_bs = bs
        datasets = [LanguageModelPreLoader(ds, shuffle=(i==0), bs=(bs if i==0 else val_bs), bptt=bptt, backwards=backwards) 
                    for i,ds in enumerate(datasets)]            
        dls = [DataLoader(d, b, shuffle=False) for d,b in zip(datasets, (bs,val_bs,val_bs,val_bs)) if d is not None]
        
        return cls(*dls, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

class BioClasDataBunch(TextClasDataBunch):
    @classmethod
    def from_folder(cls, path:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, recurse:bool=True, max_seqs_per_file:int=None,
                    valid_pct:float=0.2, train:str=None, valid:str=None, test:Optional[str]=None,
                    classes:Collection[str]=None, pad_idx:int=1, pad_first:bool=True, 
                    device:torch.device=None, backwards:bool=False, no_check:bool=False,
                    tokenizer:BioTokenizer=None, vocab:BioVocab=None, 
                    chunksize:int=10000, max_vocab:int=60000, min_freq:int=2, include_bos:bool=None, include_eos:bool=None, 
                    bs=64, val_bs:int=None, seed:int=None, **kwargs):

        path = Path(path).absolute()
        processor = get_lol_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, include_bos=include_bos, include_eos=include_eos)
        
        src = BioTextList.from_folder(path=path, extensions=extensions, recurse=recurse, max_seqs_per_file=max_seqs_per_file, processor=processor)
        if valid_pct:
            src = src.split_by_rand_pct(valid_pct=valid_pct, seed=seed)
        else:
            src = src.split_by_folder(train=train, valid=valid)
        src = src.label_from_folder(classes=classes)
        if test is not None: src.add_test_folder(path/test)

        d1 = src.databunch(**kwargs)
        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **kwargs))

        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

    @classmethod
    def from_seqfile(cls, path:PathOrStr, extensions:Collection[str]=supported_seqfiletypes, recurse:bool=True, max_seqs_per_file:int=None,
                    valid_pct:float=0.2, label_func:Callable=None,
                    classes:Collection[str]=None, pad_idx:int=1, pad_first:bool=True, backwards:bool=False,
                    device:torch.device=None, no_check:bool=False,
                    tokenizer:BioTokenizer=None, vocab:BioVocab=None, 
                    chunksize:int=10000, max_vocab:int=60000, min_freq:int=2, include_bos:bool=None, include_eos:bool=None, 
                    bs=64, val_bs:int=None, seed:int=None, **kwargs):

        "Create from a sequence file."
        path = Path(path).absolute()
        processor = get_lol_processor(tokenizer=tokenizer, vocab=vocab, chunksize=chunksize, max_vocab=max_vocab,
                                   min_freq=min_freq, include_bos=include_bos, include_eos=include_eos)

        src = BioTextList.from_seqfile(filename=filename, path=path, extensions=extensions, max_seqs_per_file=max_seqs_per_file, processor=processor)
        src = src.split_by_rand_pct(valid_pct=valid_pct, seed=seed)
        if label_func:
            src = src.label_from_func(label_func)

        if test is not None: src.add_test_folder(path/test)

        d1 = src.databunch(**kwargs)
        datasets = cls._init_ds(d1.train_ds, d1.valid_ds, d1.test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **kwargs))

        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

