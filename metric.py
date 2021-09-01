from torchtext.data.metrics import bleu_score

from PPSM_Models import translate_sentence

'''
BLEU: BLEU looks at the overlap in the predicted and actual target sequences in
terms of their n-grams. It will give us a number between 0 and 1 for each
sequence, where 1 means there is perfect overlap, i.e. a perfect translation,
although is usually shown between 0 and 100. 
BLEU was designed for multiple candidate translations per source sequence, (averaged)
however in this dataset we only have one candidate per source.

Example:
>>> from torchtext.data.metrics import bleu_score
>>> candidate_corpus = [ ['My', 'full', 'pytorch', 'test'], 
                         ['Another', 'Sentence']
                        ]
>>> references_corpus = [ [ ['My', 'full', 'pytorch', 'test'], ['Completely', 'Different'] ], 
                          [ ['No', 'Match'] ]
                        ]
>>> bleu_score(candidate_corpus, references_corpus)
    0.8408964276313782

Try out: Case sensative
>>> candidate_corpus = [ ['My', 'full', 'pytorch', 'test'], 
                         ['Another', 'Sentence']
                        ]
>>> references_corpus = [ [ ['my', 'Full', 'Pytorch', 'test'], ['completely', 'different'] ], 
                          [ ['no', 'match'] ]
                        ]
>>> bleu_score(candidate_corpus, references_corpus)

Try out: Indexes
>>> from torchtext.data.metrics import bleu_score

>>> candidate_corpus = [ ['1', '2', '3', '4'], 
                         ['5', '6']
                        ]
>>> references_corpus = [ [ ['1', '2', '3', '4'], ['7', '8'] ], 
                          [ ['9', '10'] ]
                        ]
>>> bleu_score(candidate_corpus, references_corpus)

>>> candidate_corpus = [ [1, 2, 3, 4], 
                         [5, 6]
                        ]
>>> references_corpus = [ [ [1, 2, 3, 4], [7, 8] ], 
                          [ [9, 10] ]
                        ]
>>> bleu_score(candidate_corpus, references_corpus)

By default:
max_n=4, weights=[0.25, 0.25, 0.25, 0.25].
'''

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    # buffers for all samples (as list of )
    trgs = []
    pred_trgs = []

    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg'] # a list of words

        # returns the target words list & attentions list
        # pred_trg：target sentence words list
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)

        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        # trgs.append([trg_1, trg_2, ..., trg_N]), trg_n is a list itself.
        
    return bleu_score(pred_trgs, trgs)