from collections import deque, defaultdict, Counter
import math
import nltk
import time
from collections import Counter


# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    # ummm wtf am i supposed to do here lol? 
    
    # add every word to giant list of words
    unigram = {}
    bigram = {}
    trigram = {}
    
    # unigram creation
    tokens = []
    for sentence in training_corpus:
        sentence = sentence + STOP_SYMBOL
        unigram_tokens = sentence.split()
        tokens.extend(unigram_tokens)
        # print("yourmom")
    
    # unigram_count = {word: tokens.count(word) for word in set(tokens)}
    unigram_count = Counter(tokens)
    
    total = len(tokens)
    for word in unigram_count:
        probability = math.log2(unigram_count[word]/total)
        unigram[(word,)] = probability
        # print("forever")
        
    # BIGRAM CREATION
    bigram_tokens = []
    starter = 0
    
    # have to incorporate START_SYMBOL unigram probability: #of sentences/new_total and then add it to count
    for sentence in training_corpus: 
        sentence = START_SYMBOL + " " + sentence + STOP_SYMBOL
        segment_tokens = sentence.split()
        bigram_tokens.extend(segment_tokens)
        starter += 1
    
    # add start probs to the count dict 
    unigram_count[START_SYMBOL] = starter
    bigram_tuples = list(nltk.bigrams(bigram_tokens))
    bigram_count = Counter(bigram_tuples)
    
    for word in bigram_count:  
        first = unigram_count[word[0]]
        bigram_p = math.log2(bigram_count[word]/first)
        bigram[(word)] = bigram_p
        
        
    trigram_tokens = []
    double_start = 0
    
    # have to incorporate START_SYMBOL unigram probability: #of sentences/new_total and then add it to count
    for sentence in training_corpus: 
        sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + STOP_SYMBOL
        segment_tokens = sentence.split()
        trigram_tokens.extend(segment_tokens)
        double_start += 1
    
    # add start probs to the count dict 
    bigram_count[(START_SYMBOL,START_SYMBOL)] = double_start
    
    trigram_tuples = list(nltk.trigrams(trigram_tokens))
    trigram_count = Counter(trigram_tuples)
    
    for word in trigram_count:  
        first = bigram_count[(word[0],word[1])]
        # print((word[0], word[1]))
        # print(first)
        trigram_p = math.log2(trigram_count[word]/first)
        trigram[(word)] = trigram_p
        

    return unigram, bigram, trigram

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    #output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    sorted(unigrams_keys)
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    sorted(bigrams_keys)
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    sorted(trigrams_keys)
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    
    for sequence in corpus:
        sentence = sequence + STOP_SYMBOL
        
        absent = 0
        sentence_score = 0
        if n == 1:
            unigram_list = sentence.split()
            
            for unigram in unigram_list:
                if unigram in ngram_p:
                    unigram_log = ngram_p[(unigram,)]
                    sentence_score += unigram_log
                else:
                    absent = 1
            
        elif n == 2:
            sentence = START_SYMBOL + " " + sentence
            bigram_tokens = sentence.split()
            bigram_list = list(nltk.bigrams(bigram_tokens))
            
            for bigram in bigram_list:
                if bigram in ngram_p:
                    bigram_log = ngram_p[bigram]
                    sentence_score += bigram_log
                else: 
                    absent = 1
            
        elif n == 3:
            sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence
            trigram_tokens = sentence.split()
            trigram_list = list(nltk.trigrams(trigram_tokens))
            for trigram in trigram_list:
                # add check for missing ngrams
                if trigram in ngram_p:
                    trigram_log = ngram_p[trigram]
                    sentence_score += trigram_log
                else: 
                    absent = 1
        if absent == 0: 
            scores.append(sentence_score)  
        else: 
            scores.append(MINUS_INFINITY_SENTENCE_LOG_PROB)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lamb = 1/3
    
    # get the actual probability from our log probabilities
    # using an even weighting, calculate average probabilities for each word
    # find the probability of the whole word
    # return it into a log probabikity and add it to your function
    for sentence in corpus:
        sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + STOP_SYMBOL
        trigram_tokens = list(nltk.trigrams(sentence.strip().split()))
        
        sentence_score = 0
        for trigram in trigram_tokens:
            trigram_log = trigrams.get(trigram,  MINUS_INFINITY_SENTENCE_LOG_PROB)
            bigram_log =  bigrams.get((trigram[1],trigram[2]), MINUS_INFINITY_SENTENCE_LOG_PROB)
            unigram_log = unigrams.get(trigram[2], MINUS_INFINITY_SENTENCE_LOG_PROB)
            
            sentence_prob = ((lamb * pow(2, trigram_log)) + (lamb * pow(2, bigram_log)) + (lamb * pow(2, unigram_log)) )
            sentence_log = math.log2(sentence_prob)
            sentence_score += sentence_log
            
        scores.append(sentence_score)
        
    return scores
    
# As above, but with modified lambda values
def linearscore_newlambdas(unigrams, bigrams, trigrams, corpus):
    scores = []
    
   
    l1, l2, l3 = 1/10, 5/20, 14/20
    
    # get the actual probability from our log probabilities
    # using an even weighting, calculate average probabilities for each word
    # find the probability of the whole word
    # return it into a log probabikity and add it to your function
    for sentence in corpus:
        sentence = START_SYMBOL + " " + START_SYMBOL + " " + sentence + STOP_SYMBOL
        trigram_tokens = list(nltk.trigrams(sentence.strip().split()))
        
        sentence_score = 0
        for trigram in trigram_tokens:
            trigram_log = trigrams.get(trigram,  MINUS_INFINITY_SENTENCE_LOG_PROB)
            bigram_log =  bigrams.get((trigram[1],trigram[2]), MINUS_INFINITY_SENTENCE_LOG_PROB)
            unigram_log = unigrams.get(trigram[2], MINUS_INFINITY_SENTENCE_LOG_PROB)
            
            sentence_prob = ((l3 * pow(2, trigram_log)) + (l2 * pow(2, bigram_log)) + (l1 * pow(2, unigram_log)) )
            sentence_log = math.log2(sentence_prob)
            sentence_score += sentence_log
            
        scores.append(sentence_score)
        
    return scores
    

DATA_PATH = 'data/' # path to the data
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)
    linearscores_modlambda = linearscore_newlambdas(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.reg.txt')
    score_output(linearscores_modlambda, OUTPUT_PATH + 'A3.newlambdas.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print(f"Part A time: {str(time.clock())} sec")

if __name__ == "__main__": main()
