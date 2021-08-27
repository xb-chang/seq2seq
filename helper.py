'''
Utilities from Pytorch Tutorial.
'''

import re
from io import open
import unicodedata

# representing each word in a language as a one-hot vector
# Lang class with following data;
# word2index; index2word; word2count;(replace the rare words later)

# special words
# start of sentence (SOS) token
SOS_token = 0
# end of sentence (EOS) token
EOS_token = 1

# a language object (records the word-index pair)
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('.data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    
    # Split every line into pairs (with tab '\t') and normalize
    # pairs = [[lang1_str lang2_str], ...,[lang1_str lang2_str]]
    # raw data pair saved in list
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        src_lang = Lang(lang2)
        trg_lang = Lang(lang1)
    else:
        src_lang = Lang(lang1)
        trg_lang = Lang(lang2)

    return src_lang, trg_lang, pairs

# we make the data simple: the English/French words are less than 10 (including
# the ending punctuation). We focus on the sentences start with 'I am' and 'He
# is' etc.
MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    # match the filtering requirement or not
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# The full process for preparing the data is:
# Read text file and split into lines, split lines into pairs
# Normalize text, filter by length and content
# Make word lists from sentences in pairs
def prepareData(lang1, lang2, reverse=False):
    src_lang, tar_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    # counting on the filtered data.
    print("Counting words...")
    for pair in pairs:
        src_lang.addSentence(pair[0])
        tar_lang.addSentence(pair[1])
    print("Counted words:")
    print(src_lang.name, src_lang.n_words)
    print(tar_lang.name, tar_lang.n_words)

    return src_lang, tar_lang, pairs
