#data processing: tokenizing, creating vocabs, etc

from fastai.torch_core import *
from fastai.data_block import *
from fastai.text import Vocab
import itertools

supported_languages = {'dna':['A','C','G','T']}
supported_seqfiletypes = ['.fastq']

UNK, PAD = 'xxunk', 'xxpad'
defaults.special_tokens = [UNK, PAD]

class BioTokenizer():
    """
    tokenize text with multiprocessing. 
    
    Note in fastai.text there is a separate BaseTokenizer class to contain the tokenizer function and that is passed to the 'Tokenizer' class. Here, we put this in one class to make this less confusing.
    """

    def __init__(self, ksize:int=1, stride:int=1, special_cases:Collection[str]=None, alphabet:str='dna', n_cpus:int=None):
        self.ksize = ksize #the kmer size (or 'ngram') of each token (default character level k=1)
        self.stride = stride #length of stride to take between each kmer
        self.alphabet = alphabet
        self.special_cases = special_cases
        self.n_cpus = ifnone(n_cpus, defaults.cpus)

    def __repr__(self) -> str:
        res = f'BioTokenizer with the {self.alphabet} alphabet with the following special cases:\n'
        for case in self.special_cases:
            res += f' - {case}\n'
        return res

    def tokenizer(self, t:str) -> List[str]:
        t = t.upper()
        if self.ksize == 1:
            toks = list(t)
        else:
            toks = [t[i:i+self.ksize] for i in range(0, len(t), self.stride) if len(t[i:i+self.ksize]) == self.ksize]
        if len(toks[-1]) < self.ksize: 
            toks = toks[:-1]

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
        "Create a vocabulary from a set of `tokens`."
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(defaults.special_tokens):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        itos = itos[:max_vocab]
        return cls(itos)

    @classmethod
    def create_from_ksize(cls, ksize:int=1, alphabet:Collection[str]=supported_languages['dna'], special_tokens=defaults.special_tokens):
        "Create a vocabulary of all possible permutations of kmers of a given kmer size. If no "
        itos = kmer_permutations(ksize=ksize, alphabet=alphabet)
        for o in reversed(defaults.special_tokens):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        return cls(itos)

#The following transforms are what you need to apply to items in each batch - open the sequence, tokenize, and numericalize
def OpenSeq(item, tokenizer:BioTokenizer=BioTokenizer(), 
                vocab:BioVocab=BioVocab.create_from_ksize(), 
                sep:str=' ', 
                extensions:Collection[str]=supported_seqfiletypes):

    filename, offset = item
    seq = open_single_read(filename, offset, tok=tokenizer, vocab=vocab, sep=sep, extensions=extensions)
    return seq

#def TokenizeSeq(item, )


#The following PreProcessors can be applied to preprocess items in your SeqList (which are pointers to sequences in files by default) into numericalized tokens before training. 
# These preprocessors will convert each pointer in the itemlist to a tokenized & numericalized text. Note this will mean all your items needs to fit in memory

def _join_seqs(seqs:Collection[str]):
    #if your sequence is an array of multiple sequences, this will join them together separated by spaces to be tokenized as one
    #(this may be useful if for example you want to represent a genome from its genes in syntenic order perhaps with a special token to mark where genes begin and end)
    
    if not isinstance(seqs, np.ndarray): seqs = np.array(seqs)
    if is1d(seqs): seqs = seqs[:,None]
    df = pd.DataFrame({i:seqs[:,i] for i in range(seqs.shape[1])})
    col = '' + df[0].astype(str)
    for i in range(1,len(df.columns)):
        col += ' ' + df[i].astype(str)   
    return col.values

class OpenSequenceProcessor(PreProcessor):
    def __init__(self, ds:ItemList=None, tokenizer:BioTokenizer=BioTokenizer(), vocab:BioVocab=BioVocab.create_from_ksize(), sep:str=' ', extensions:Collection[str]=supported_seqfiletypes):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.extensions = extensions
        self.sep = sep

    def process_one(self, item):
        filename, offset = item
        seq = open_single_read(filename, offset, tok=self.tokenizer, vocab=self.vocab, sep=self.sep, extensions=self.extensions)
        return seq

    def process(self, ds):
        #ds.items =  array([self.process_one(item) for item in ds.items], dtype=np.object)
        opened_items =  array([self.process_one(item) for item in ds.items], dtype=np.object)
        return opened_items

class BioTokenizeProcessor(PreProcessor):
    "`PreProcessor` that tokenizes the seqs in `ds`."
    def __init__(self, ds:Collection=None, tokenizer:BioTokenizer=BioTokenizer(), chunksize:int=10000):
        self.tokenizer = tokenizer
        self.chunksize = chunksize

    def process_one(self, item):
        return self.tokenizer._process_all_1(_join_seqs([item]))[0]

    def process(self, ds):
        ds.items = _join_seqs(ds.items)
        tokens = []
        for i in progress_bar(range(0,len(ds),self.chunksize), leave=False):
            tokens += self.tokenizer.process_all(ds.items[i:i+self.chunksize])
        ds.items = tokens

#class BioNumericalizeProcessor(PreProcessor)

def get_lol_processor(tokenizer:BioTokenizer=BioTokenizer(), vocab:BioVocab=BioVocab.create_from_ksize(), 
                        extensions:Collection[str]=None, 
                        max_vocab:int=4**8, min_freq:int=1):
    #see Chor et al. 2009 (doi: 10.1186/gb-2009-10-10-r108) for kmer frequency distribs for diff k sizes that make sense

    return [OpenSequenceProcessor(tokenizer=tokenizer, vocab=vocab, extensions=extensions),
            BioTokenizeProcessor(tokenizer=tokenizer), 
            BioNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

