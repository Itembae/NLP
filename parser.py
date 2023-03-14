import sys
import nltk
import math
import time
from collections import defaultdict
from collections import deque
import heapq
from collections import Counter
import collections

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    # list of lists
    brown_words = []
    brown_tags = []

    for sentence in brown_train:
        tokens = sentence.split()
        sentence_words = []
        sentence_tags = []
        for token in tokens:
            split = token.rsplit("/", 1)
            word = split[0]
            tag = split[1]
            sentence_words.append(word)
            sentence_tags.append(tag)
            
        sentence_tags.append(STOP_SYMBOL)
        sentence_tags.insert(0, START_SYMBOL)
        sentence_tags.insert(0, START_SYMBOL)
        sentence_words.append(STOP_SYMBOL)
        sentence_words.insert(0, START_SYMBOL)
        sentence_words.insert(0, START_SYMBOL)
        
        brown_tags.append(sentence_tags)
        brown_words.append(sentence_words)
        # print(brown_words[0])
    return brown_words, brown_tags


# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    
    trigram = {}
    bigram_tokens = []
    for sentence in brown_tags: 
        bigram_tokens.extend(sentence)
    bigram_tuples = list(nltk.bigrams(bigram_tokens))
    bigram_count = Counter(bigram_tuples)
        
            # TRIGAM
    trigram_tokens = []

    for sentence in brown_tags: 
        segment_tokens = sentence
        trigram_tokens.extend(segment_tokens)
    
    trigram_tuples = list(nltk.trigrams(trigram_tokens))
    trigram_count = Counter(trigram_tuples)
    
    for word in trigram_count:  
        first = bigram_count[(word[0],word[1])]
        trigram_p = math.log2(trigram_count[word]/first)
        trigram[(word)] = trigram_p
        
    q_values = trigram
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!

# Checked
def calc_known(brown_words):
    known_words = set([])
    counter = Counter()
    # print(brown_words[0])
    for words in brown_words:
        counter.update(words)
    for word in counter:
            if counter[word] > RARE_WORD_MAX_FREQ:
                known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
# checked
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for words in brown_words:
        word_list = []
        count = 0
        for word in words:
            temp = word
            if word not in known_words: 
                temp = RARE_SYMBOL 
            word_list.append(temp)
            count+=1
        brown_words_rare.append(word_list)
    return brown_words_rare



# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set (should not include start and stop tags)
def calc_emission(brown_words_rare, brown_tags):
    
    # emission probability is the probability of a word given a particular tag
    # tally up occurrences of the word with that tag
    # tally up the occurrences of that particular tag
    
    e_values = {}
    taglist = set([])
    
    assert(len(brown_words_rare) == len(brown_tags))
    
    corpus_tuples = []
    tags = []
    for sentence in range(len(brown_words_rare)): 
        for word in range(len(brown_words_rare[sentence])):
            corpus_tuples.append((brown_words_rare[sentence][word], brown_tags[sentence][word]))
            tags.append(brown_tags[sentence][word])
    
    tuple_counts = Counter(corpus_tuples)
    tag_counts = Counter(tags)
    
    # print("format",tuple_counts)
    
    for tuple in tuple_counts:
        tup_p = tuple_counts[tuple]/(tag_counts[tuple[1]])
        tup_log = math.log2(tup_p)
        e_values[tuple] = tup_log
        
    taglist = set(tags)
    taglist.discard(START_SYMBOL)
    taglist.discard(STOP_SYMBOL)
    return e_values, taglist
    

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    sorted(emissions)
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!


# def prev_word_n(sentence, t, n):
#     if t - n < 0: return START_SYMBOL
#     return sentence[t - n]

# def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
#     tagged = []
    
#     # initialization
#     viterbi_matrix = {}
#     backpointer = {}
#     viterbi_matrix[(-1, START_SYMBOL, START_SYMBOL,)] = 0.0
#     backpointer[(-1, START_SYMBOL, START_SYMBOL,)] = START_SYMBOL
 
#     possible_tags = defaultdict(list)
#     for word, tag in itertools.product(known_words, taglist):
#         try:
#             p = e_values[(word, tag,)]
#             possible_tags[word].append(tag)
#         except KeyError: continue
        
#     possible_tags[RARE_SYMBOL] = [tag for tag in taglist]
#     possible_tags[START_SYMBOL].append(START_SYMBOL)
#     possible_tags[STOP_SYMBOL].append(STOP_SYMBOL)
 
#     for sentence in brown_dev_words:
#         tokens = []
#         for word in sentence:
#             if word in known_words: tokens.append(word)
#             else: tokens.append(RARE_SYMBOL)
#         tokens_len = len(tokens)

#         for t in range(tokens_len):
#             word = prev_word_n(tokens, t, 0)
#             word_p = prev_word_n(tokens, t, 1)
#             word_p_p = prev_word_n(tokens, t, 2)
            
#             # curr state and prev state we're considering
#             for state, state_p in itertools.product(possible_tags[word], possible_tags[word_p]):
#                     max_prob = float('-inf')
#                     max_state_p_p = None
                    
#                     # prev prev state
#                     for state_p_p in possible_tags[word_p_p]:
#                         a = q_values.get((state_p_p, state_p, state,), LOG_PROB_OF_ZERO)
#                         b = e_values.get((word, state,), LOG_PROB_OF_ZERO)

#                         state_p_t_minus_one = viterbi_matrix.get((t-1, state_p_p, state_p,), LOG_PROB_OF_ZERO)
          
#                         prob = state_p_t_minus_one + a + b
#                         if prob > max_prob:
#                             max_prob = prob
#                             max_state_p_p = state_p_p
#                     viterbi_matrix[(t, state_p, state,)] = max_prob
#                     backpointer[(t, state_p, state,)] = max_state_p_p
        
#         # find best prob from STOP
#         max_prob = float('-inf')
#         max_state_p = None
#         max_state_p_p = None
        
#         last_idx = tokens_len - 1
#         word_p = prev_word_n(tokens, last_idx, 0)
#         word_p_p = prev_word_n(tokens, last_idx, 1)
#         for state_p, state_p_p in itertools.product(possible_tags[word_p], possible_tags[word_p_p]):
#                 a = q_values.get((state_p_p, state_p, STOP_SYMBOL,), LOG_PROB_OF_ZERO)
#                 prob = viterbi_matrix.get((last_idx, state_p_p, state_p,), LOG_PROB_OF_ZERO) + a
                
#                 if prob > max_prob:
#                     max_prob = prob
#                     max_state_p = state_p
#                     max_state_p_p = state_p_p

#         # backtrack to get tags
#         count = 0
#         tags = [max_state_p, max_state_p_p]
#         for t in range(tokens_len - 1, 1, -1):
#             tag = backpointer[(t, tags[count + 1], tags[count],)]
#             tags.append(tag)
#             count += 1
        
#         # pair word w/ their tag and add to tagged list
#         tagged_sentence = []
#         for word, tag in zip(sentence, tags[::-1]):
#             tagged_sentence.append(word + '/' + tag)

#         tagged_sentence.append("\n")
#         tagged.append(" ".join(tagged_sentence))

#     return tagged
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    
    for sentence in brown_dev_words:
        #store the probability of a given tag sequence ending in the tag that is also the key
        viterbi_matrix = collections.defaultdict(float)
        # stores the most likely sequence that gave rise to a particular tag
        new_paths = {}
    
        # copy over sentence to prevent accidental modification of source
        tokens = [] 
        for word in sentence:
            if word not in known_words:
                tokens.append(RARE_SYMBOL)
            else:
                tokens.append(word)
                
        for tag in taglist:
            word = tokens[0]
            emit_tuple = (word, tag)
            # only consider it if it has an emission value 
            if emit_tuple in e_values:
                likeliness = q_values.get((START_SYMBOL, START_SYMBOL, tag), LOG_PROB_OF_ZERO) + e_values[emit_tuple]
                new_paths[tag] = [START_SYMBOL, START_SYMBOL, tag]
                viterbi_matrix[tag] = likeliness
            else:
                viterbi_matrix[tag] = 0
        
        
        
        length = len(sentence)
        for word in range(1, length):
            # update
            old_paths = {}
            old_paths = new_paths.copy()
            
            # manually copy over dict in case dict.copy() is responsible for randomization 
            
            # for key in new_paths:
            #     old_paths[key] = []
            #     for item in new_paths[key]:
            #         old_paths[key].append(item)
                    
            new_paths.clear()
            token = tokens[word]
            valid_tags = []
            
            # only checks tags that have a positive emission probability for the current word
            if token == RARE_SYMBOL: 
                valid_tags = taglist
            else:
                for tags in taglist:
                    if (token, tags) in e_values:
                        valid_tags.append(tags)
                        
                   
            # calculates viterbi probability for each tag at the given word 
            for tag in valid_tags:
                # second (redundant) check to make sure we only consider positive e_values 
                if (token, tag) in e_values:
                    max = float('-Inf')
                    best_path = None
                    # old_paths contains all valid tag sequences for the n-1 preceding words
                    for path in taglist:
                        if path in old_paths:
                            prob = viterbi_matrix.get(path, LOG_PROB_OF_ZERO) + q_values.get((old_paths[path][-2], path, tag), LOG_PROB_OF_ZERO) + e_values[(token, tag)]
                            if prob > max:
                                max = prob
                                best_path = path
                    viterbi_matrix[tag] = max
                    
                    new_paths[tag] = old_paths[best_path].copy()
                    new_paths[tag].append(tag)
                    
                    # patch for dict.copy()
                    # new_paths[tag] = []
                    # for item in old_paths[best_path]:
                    #     new_paths[tag].append(item)

                    
                # reset the probability for the next word having this tag as zero 
                else:
                    viterbi_matrix[tag] = 0
                    
        max = -10000
        best_sequence = []
        
        # check valid tags for the final word and select the one with highest sequence probability
        for path in taglist:
            if path in new_paths:
                if viterbi_matrix[path] > max:
                    max = viterbi_matrix[path]
                    best_sequence = new_paths[path]
        
        # remove START_SYMBOLS from  the selected path to be assigned to the sentence
        if START_SYMBOL in best_sequence:
            best_sequence.remove(START_SYMBOL)
        if START_SYMBOL in best_sequence:
            best_sequence.remove(START_SYMBOL)
        
        
        output_sentence = ""
        assert(len(sentence) == len(best_sequence))
        for i in range(length):
            entry = sentence[i] + "/" + best_sequence[i] 
            output_sentence += entry + " "
        
        output_sentence += "\n"
            
        tagged.append(output_sentence)
    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in range(len(brown_words)) ]
    training = [list(x) for x in training]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff = default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff = bigram_tagger)
    for sentence in brown_dev_words:
        tagged.append(' '.join([word + '/' + tag for word, tag in trigram_tagger.tag(sentence)]) + '\n')
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/' # path to the data
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print(f"Part B time: {str(time.clock())} sec")

if __name__ == "__main__": main()
