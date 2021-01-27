import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot, Tokenizer

class decoder_tokenizer:

    def __init__(self, MAX_LEN=20):
        self.MAX_LEN = MAX_LEN
        
    def fit(self, list_of_docs):
        """
        This func leverages pandas, 
        so the input is pandas series of output datetime strings
        
        :param series_of_docs: pd.Series of datetime strings, %d/%m/%Y
        """
        # generate vocab
        sequenced_docs = [text_to_word_sequence(self.preprocess(doc), filters='') for doc in list_of_docs]
        MAX_LEN = max([len(sequenced_doc) for sequenced_doc in sequenced_docs])
        print(f"max len: {MAX_LEN}")
        self.MAX_LEN = MAX_LEN + 10
        decodable_vocab = [element for list_ in sequenced_docs for element in list_]
        decodable_vocab = set(decodable_vocab)
        decodable_vocab.add( 'EOS' )
        decodable_vocab.add( 'SOS' )
        
        # fit tokenizer
        self.t = Tokenizer(filters='')
        self.t.fit_on_texts(list(set(decodable_vocab)))
        self.mapping = self.t.word_index
        self.vocab_size = len(self.mapping) + 1
        
    def tokenize(self, string_dates, sos=True, eos=True):
        """
        Takes in list of strings
        """
        string_dates = [self.preprocess(string_date, sos=sos, eos=eos) 
                        for string_date in string_dates]
        tokenized = self.t.texts_to_sequences(string_dates)
        
        padded = pad_sequences(tokenized, maxlen=self.MAX_LEN, padding='post',truncating='post')
        return padded
    
    def detokenize(self, seq):
        """
        t.detokenize([[53, 20, 49, 47, 49, 27, 66]])
        >>> ['sos 25 / 12 / 2006 eos']
        """
        detokenized = self.t.sequences_to_texts(seq)
        return detokenized
    
    def _add_SOS_EOS(self, string, sos=True, eos=True):
        if sos: string = 'SOS ' + string
        if eos: string = string + ' EOS'
        return string
    
    def preprocess(self, string_date, sos=True, eos=True):
        """
        include any further pre-tokenizing steps here
        """
        string_date = string_date.replace('/', ' / ')
        string_date = self._add_SOS_EOS(string_date, sos=sos, eos=eos)
        return string_date

    def save(self, path='tokenizer.json'):
        # https://stackoverflow.com/questions/45735070/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring
        tokenizer_json = self.t.to_json()
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
            
    def load(self, path='tokenizer.json'):
        with open(path) as f:
            data = json.load(f)
            self.t = keras.preprocessing.text.tokenizer_from_json(data)
        
        # store impt details
        self.mapping = self.t.word_index
        self.vocab_size = len(self.mapping) + 1