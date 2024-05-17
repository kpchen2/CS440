"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    dict = {}
    tag_dict = {}
    
    for sentence in train:
        for pair in sentence:
            cur_word = pair[0]
            cur_tag = pair[1]

            if (cur_word in dict.keys()):
                if (cur_tag in dict[cur_word].keys()):
                    dict[cur_word][cur_tag] += 1
                else:
                    dict[cur_word][cur_tag] = 1
            else:
                dict[cur_word] = {}
                dict[cur_word][cur_tag] = 1

            if (cur_tag in tag_dict.keys()):
                tag_dict[cur_tag] += 1
            else:
                tag_dict[cur_tag] = 1

    output = []
    for sentence in test:
        cur = []
        for word in sentence:
            if word in dict.keys():
                tag = max(dict[word], key=dict[word].get)
                cur.append((word, tag))
            else:
                tag = max(tag_dict, key=tag_dict.get)
                cur.append((word, tag))
        output.append(cur)
    return output