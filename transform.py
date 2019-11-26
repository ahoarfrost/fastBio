#data processing: tokenizing, creating vocabs, etc

from fastai import *
from fastai.text import *
import itertools

supported_languages = {'dna':['A','C','G','T']}

UNK, PAD, BOS, EOS = 'xxunk', 'xxpad', 'xxbos', 'xxeos'
defaults.special_tokens = [UNK, PAD, BOS, EOS]

class BioTokenizer():
    """
    tokenize text with multiprocessing. 
    
    Note in fastai.text there is a separate BaseTokenizer class to contain the tokenizer function and that is passed to the 'Tokenizer' class. Here, we put this in one class to make this less confusing.
    """

    def __init__(self, ksize:int=1, stride:int=1, special_cases:Collection[str]=None, n_cpus:int=None, 
                 include_bos:bool=True, include_eos:bool=False):
        self.ksize = ksize #the kmer size (or 'ngram') of each token (default character level k=1)
        self.stride = stride #length of stride to take between each kmer
        self.special_cases = special_cases if special_cases is not None else defaults.special_tokens
        self.n_cpus = ifnone(n_cpus, defaults.cpus)
        self.include_bos, self.include_eos = include_bos, include_eos

    def __repr__(self) -> str:
        res = f'BioTokenizer with the following special tokens:\n'
        for case in self.special_cases:
            res += f' - {case}\n'
        return res

    def tokenizer(self, t:str, include_bos:bool=None, include_eos:bool=None) -> List[str]:
        include_bos = ifnone(include_bos, self.include_bos)
        include_eos = ifnone(include_eos, self.include_eos)

        t = t.upper()
        if self.ksize == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i+self.ksize]) == self.ksize]
        if len(toks[-1]) < self.ksize: 
            toks = toks[:-1]

        if include_bos:
            toks = [f'{BOS}'] + toks
        if include_eos:
            toks = toks + [f'{EOS}']

        return toks

    def add_special_cases(self, toks):
        #TODO: add e.g. padding variable
        pass

    def process_one(self, s:str) -> List[str]:
        "Process one sequence `s` with tokenizer `tok`."
        toks = self.tokenizer(s)
        return toks

    def _process_all_1(self, seqs:Collection[str]) -> List[List[str]]:
        "Process a list of `seqs` in one process."
        if self.special_cases: self.add_special_cases(self.special_cases)
        return [self.process_one(str(s)) for s in seqs]

    def process_all(self, seqs:Collection[str]) -> List[List[str]]:
        "Process a list of `seqs` with multiprocessing."
        if self.n_cpus <= 1: return self._process_all_1(seqs)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(seqs, self.n_cpus)), [])

def kmer_permutations(ksize, alphabet):
    '''
    This function returns the set of all possible permutations of letters/nucleotides for a given kmer size.
    alphabet: a list of the possible letters in your alphabet; e.g. for DNA this is ['A','C','G','T']
    k_size: an integer, the kmer size of your resulting permutations; e.g. 6
    The number of resulting permutations will be alphabet**k_size; e.g. for 10-mers of DNA nucleotides, there are 4^10 or 1048576 possible combinations of nucleotides.
    '''
    permutations = set()
    for combo in itertools.product(alphabet, repeat=ksize):
        permutations.add(''.join(combo))
    return list(permutations)

class BioVocab(Vocab):
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos:Collection[str]):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    @classmethod
    def create(cls, tokens:Collection[str]=None, special_tokens=defaults.special_tokens, max_vocab:int=4**8, min_freq:int=100) -> 'Vocab':
        '''
        Create a vocabulary from a set of `tokens` based on a maximum vocabulary size and minimum frequency of the occurrence of the token.
        Defaults will fit 4**8 tokens (so e.g. all permutations of ksize=8), with a minimum frequency of 100.
        see e.g. Chor et al. 2009 (doi: 10.1186/gb-2009-10-10-r108) for kmer frequency distribs for diff k sizes that make sense
        '''
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(special_tokens):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        itos = itos[:max_vocab]
        return cls(itos)

    @classmethod
    def create_from_ksize(cls, ksize:int=1, alphabet:Collection[str]=supported_languages['dna'], special_tokens=defaults.special_tokens):
        '''
        Create a vocabulary of all possible permutations of kmers of a given kmer size. 
        In DNA, this remains practical up until a ksize of about 8 (4**8=65536).
        '''
        itos = kmer_permutations(ksize=ksize, alphabet=alphabet)
        for o in reversed(special_tokens):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        return cls(itos)

#The following PreProcessors can be applied to preprocess items in your TextList into numericalized tokens before training. 
# These preprocessors will convert each sequence (in Text form) in the itemlist to a tokenized & numericalized text. Note this will mean all your items need to fit in memory

def _join_seqs(seqs:Collection[str]):
    '''
    if your sequence is an array of multiple sequences, this will join them together separated by spaces to be tokenized as one
    (this may be useful if for example you want to represent a genome from its genes in syntenic order perhaps with a special token to mark where genes begin and end)
    It also adds BOS and/or EOS tokens.
    '''
    
    if not isinstance(seqs, np.ndarray): seqs = np.array(seqs)
    if is1d(seqs): seqs = seqs[:,None]

    df = pd.DataFrame({i:seqs[:,i] for i in range(seqs.shape[1])})
    col = df[0].astype(str)
    for i in range(1,len(df.columns)):
        col += ' ' + df[i].astype(str)   
    
    return col.values

class BioTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the seqs in `ds`. If no tokenizer given, defaults to character LM."
    def __init__(self, ds:ItemList=None, tokenizer:BioTokenizer=BioTokenizer(), 
                 chunksize:int=10000, include_bos:bool=None, include_eos:bool=None):
        self.tokenizer = tokenizer
        self.chunksize = chunksize
        self.include_bos = include_bos
        self.include_eos = include_eos

        if self.include_bos is not None:
            self.tokenizer.include_bos = self.include_bos 
        if self.include_eos is not None:
            self.tokenizer.include_eos = self.include_eos

    def process_one(self, item):
        return self.tokenizer._process_all_1(_join_seqs([item]))[0]

    def process(self, ds):
        ds.items = _join_seqs(ds.items)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

class BioNumericalizeProcessor(PreProcessor):
    "`PreProcessor` that numericalizes the tokens in `ds`. If no vocab given, defaults to creating one from existing tokens."
    def __init__(self, ds:ItemList=None, vocab:Vocab=None, max_vocab:int=4**8, min_freq:int=2):
        vocab = ifnone(vocab, ds.vocab if ds is not None else None)
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def process_one(self,item): return np.array(self.vocab.numericalize(item), dtype=np.int64)

    def process(self, ds):
        if self.vocab is None: 
            self.vocab = BioVocab.create(ds.items, self.max_vocab, self.min_freq)
        ds.vocab = self.vocab
        super().process(ds)


def get_lol_processor(tokenizer:BioTokenizer=None, vocab:BioVocab=None, 
                        chunksize:int=10000, max_vocab:int=4**8, min_freq:int=2,
                        include_bos:bool=None, include_eos:bool=None):

    return [BioTokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, include_bos=include_bos, include_eos=include_eos), 
            BioNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

