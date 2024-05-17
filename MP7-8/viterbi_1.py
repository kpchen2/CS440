"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    
    # return dicts
    init_prob = {}
    emit_prob = {}
    trans_prob = {}

    # helper variables
    tt_dict = {}
    wt_dict = {}
    fw_dict = {}
    tags = []
    words = []

    # process training data
    for s in sentences:   # loop through sentences
        first_word = True

        for pair in s:   # loop through words
            cur_word = pair[0]
            cur_tag = pair[1]

            # populate tags list
            if (cur_tag not in tags):
                tags.append(cur_tag)

            # populate words list
            if (cur_word not in words):
                words.append(cur_word)

            # populate word given tag dict
            if (cur_tag in wt_dict.keys()):
                if (cur_word in wt_dict[cur_tag]):
                    wt_dict[cur_tag][cur_word] += 1
                else:
                    wt_dict[cur_tag][cur_word] = 1
            else:
                wt_dict[cur_tag] = {}
                wt_dict[cur_tag][cur_word] = 1

            # populate tag given previous tag dict
            if not first_word:
                if (prev_tag in tt_dict.keys()):
                    if (cur_tag in tt_dict[prev_tag]):
                        tt_dict[prev_tag][cur_tag] += 1
                    else:
                        tt_dict[prev_tag][cur_tag] = 1
                else:
                    tt_dict[prev_tag] = {}
                    tt_dict[prev_tag][cur_tag] = 1
            prev_tag = cur_tag

            # populate first word dict
            if (first_word):
                if (cur_tag in fw_dict.keys()):
                    fw_dict[cur_tag] += 1
                else:
                    fw_dict[cur_tag] = 1
                first_word = False

    # populate return dicts
    for t in tags:

        # populate init_prob dict
        value_sum = sum(fw_dict.values())
        if (t in fw_dict.keys()):
            init_prob[t] = fw_dict[t]/value_sum

        # populate emit_prob dict
        temp_dict = {}
        v_t = len(wt_dict[t].keys())
        n_t = sum(wt_dict[t].values())
        for word in words:
            if (word in wt_dict[t]):
                temp_dict[word] = (wt_dict[t][word] + emit_epsilon)/(n_t + emit_epsilon*(v_t+1))
        temp_dict["UNKNOWN"] = (emit_epsilon)/(n_t + emit_epsilon*(v_t+1))
        emit_prob[t] = temp_dict

        # populate trans_prob dict
        if (t in tt_dict.keys()):
            temp_dict = {}
            v_t = len(tt_dict[t].keys())
            n_t = sum(tt_dict[t].values())
            for tag in tags:
                if (tag in tt_dict[t]):
                    temp_dict[tag] = (tt_dict[t][tag] + emit_epsilon)/(n_t + emit_epsilon*(v_t+1))
                else:
                    temp_dict[tag] = (emit_epsilon)/(n_t + emit_epsilon*(v_t+1))
            temp_dict["UNKNOWN"] = (emit_epsilon)/(n_t + emit_epsilon*(v_t+1))
            trans_prob[t] = temp_dict

    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    if i == 0:
        return prev_prob, prev_predict_tag_seq

    for tb in emit_prob:   # loops through current col
        # if (tb == "START"):
        #     continue
        # populates log_prob
        m_tag = ""
        m_val = -999999999

        # save word
        save_word = word
        
        if (word not in emit_prob[tb]):
            word = "UNKNOWN"

        for ta in emit_prob:   # loops through previous col
            if (ta == "END"):
                continue
            
            if (prev_prob[ta] + log(trans_prob[ta][tb]) + log(emit_prob[tb][word]) > m_val):
                m_val = prev_prob[ta] + log(trans_prob[ta][tb]) + log(emit_prob[tb][word])
                m_tag = ta
        prev = m_tag

        log_prob[tb] = m_val

        # populates predict_tag_seq
        prev_predict_tag_seq[(i, tb)] = prev

        # restore word
        word = save_word

    return log_prob, prev_predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)

    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[(0,t)] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)

        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        findLength = []
        for t in emit_prob:
            findLength.append((log_prob[t],t))
        end_tag = max(findLength)[1]

        path = []
        for i in range(length):
            path.append(end_tag)
            end_tag = predict_tag_seq[(length-i-1, end_tag)]
        path.reverse()
        
        p = []
        for i in range(length):
            p.append((sentence[i],path[i]))
        predicts.append(p)
    
    return predicts




