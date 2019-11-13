#data processing: tokenizing, creating vocabs, etc

from fastai.torch_core import *
from fastai.text import Vocab

supported_languages = {'dna':['A','C','G','T']}

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
        return [self.process_text(str(s)) for s in seqs]

    def process_all(self, seqs:Collection[str]) -> List[List[str]]:
        "Process a list of `seqs` with multiprocessing."
        if self.n_cpus <= 1: return self._process_all_1(seqs)
        with ProcessPoolExecutor(self.n_cpus) as e:
            return sum(e.map(self._process_all_1, partition_by_cores(seqs, self.n_cpus)), [])

class BioVocab(Vocab):
    "Contain the correspondence between numbers and tokens and numericalize."
    def __init__(self, itos:Collection[str]):
        self.itos = itos
        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})

    @classmethod
    def create(cls, tokens:Collection[str]=supported_languages['dna'], max_vocab:int=4**8, min_freq:int=1) -> 'Vocab':
        "Create a vocabulary from a set of `tokens`."
        freq = Counter(p for o in tokens for p in o)
        itos = [o for o,c in freq.most_common(max_vocab) if c >= min_freq]
        for o in reversed(defaults.special_tokens):
            if o in itos: itos.remove(o)
            itos.insert(0, o)
        itos = itos[:max_vocab]
        return cls(itos)
    
def get_processor(tokenizer:BioTokenizer=None, vocab:BioVocab=None, max_vocab:int=4**8, min_freq:int=100):
    #see Chor et al. 2009 (doi: 10.1186/gb-2009-10-10-r108) for kmer frequency distribs for diff k sizes that make sense

    return [BioTokenizeProcessor(tokenizer=tokenizer), 
            BioNumericalizeProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

