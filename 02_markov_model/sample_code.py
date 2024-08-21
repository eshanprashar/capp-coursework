from markovian.collections.lp_hashtable import LPHashtable
import math

HASH_CELLS = 57
TOO_FULL = 0.5
GROWTH_RATIO = 2

class Markov:
    def __init__(self, k, text):
        """
        Construct a new k-order markov model using the text 'text'.
        """
        self.k = k
        self.text = text
        self.markov_hashtable = LPHashtable(HASH_CELLS,0)

    def generate_string_lst (self,text):
        string_lst = []
        for i in range(len(text)):
            counter = 0
            k_gram = text[i]
            k_nxt_gram = k_gram
            nxt_index = i

            if self.k == 1:
                second_nxt_index = nxt_index + 1  
                if second_nxt_index >= len(text): #potential to make this modular 
                    second_nxt_index = 0
                k_nxt_gram = k_gram + text[second_nxt_index]
            else:
                while counter < (self.k)-1: 
                    nxt_index += 1
                    if nxt_index >= len(text):
                        nxt_index = 0
                    k_gram = k_gram + text[nxt_index]

                    second_nxt_index = nxt_index + 1  
                    if second_nxt_index >= len(text): #potential to make this modular 
                        second_nxt_index = 0
                    k_nxt_gram = k_gram + text[second_nxt_index]
                    counter += 1
            string_lst.append(k_gram)
            string_lst.append(k_nxt_gram)
        print(string_lst)
        return string_lst

    def build_markov_model(self):
        
        string_lst = self.generate_string_lst(self.text)
        for markov_key in string_lst:
            if self.markov_hashtable[markov_key] == 0:
                key_val = 1
            else:
                key_val = self.markov_hashtable[markov_key] + 1
            self.markov_hashtable[markov_key] = key_val
        return self.markov_hashtable  

    def log_probability(self, s):
        """
        Get the log probability of string "s", given the statistics of
        character sequences modeled by this particular Markov model
        This probability is *not* normalized by the length of the string.
        """
        self.s = s
        m_model = self.build_markov_model() 
        test_string_lst = self.generate_string_lst(s)
        s_value = len(set(self.text))
        character_prob = 0
        for idx, val in enumerate(test_string_lst):
            if idx % 2 != 0:
                continue
            k_element = val 
            next_k_element = test_string_lst[idx+1]
            m_value = m_model[next_k_element]
            n_value = m_model[k_element]
            character_prob = character_prob + math.log((m_value+1)/(n_value + s_value))
        return character_prob