import random, string, datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import tensorflow_text

from decoder_tokenizer import decoder_tokenizer

"""
Utils
"""
def apply_noisy_insertion(noiseless_string, noise):
    assert noise < 1
    if random.uniform(0,1) < noise:
        inserted_letter = random.choice(string.ascii_letters)
        insertion_position = np.random.randint(len(noiseless_string))
        return noiseless_string[:insertion_position] + inserted_letter + noiseless_string[insertion_position:]
    else:
        return noiseless_string

def parse_decodable_str(decodable_str):
    """
    start, end = parse_decodable_str('26 / 01 / 2012 - 27 / 06 / 2016 eos')
    start, end
    >>> (datetime.datetime(2012, 1, 26, 0, 0), datetime.datetime(2012, 1, 26, 0, 0))
    """
    decodable_str = decodable_str.replace(' eos', '')
    
    split_decodable_str = decodable_str.split(' - ')
    if len(split_decodable_str) == 1:
        return datetime.datetime.strptime('26 / 01 / 2012', '%d / %m / %Y'), None
    elif len(split_decodable_str) == 2:
        return parse_decodable_str(split_decodable_str[0])[0], parse_decodable_str(split_decodable_str[1])[0]

"""
Data
"""
class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, df, tokenizer, noise=0.4,
                 batch_size=32, shuffle=True):
        self.df = df
        self.natural_string_input = df.date_string
        self.output_string = df.decodable_string
        self.indexes = df.index.values
        self.batch_size = batch_size
        self.noise = noise
        self.shuffle = shuffle
        
        self.tokenizer = tokenizer
        self.n_classes=tokenizer.vocab_size
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.indexes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # PREPARE TRAINING DATA
        # natural string dates
        natural_date_str_batch = self.df.loc[indexes].date_string.copy()
        if self.noise>0: natural_date_str_batch = natural_date_str_batch.apply(lambda x: apply_noisy_insertion(x, self.noise))
        natural_date_str_batch = natural_date_str_batch.values
        
        # decodable strings, these are the desired output
        decodable_string_batch = self.df.loc[indexes].decodable_string

        # teacher forcing inputs
        decoder_inputs_array = self.tokenizer.tokenize(decodable_string_batch)
        decoder_inputs_array = to_categorical(
            decoder_inputs_array, num_classes=self.n_classes, dtype='float32'
        )

        # true labels, lags teacher forcing inputs by 1 time step
        decoder_outputs_array = self.tokenizer.tokenize(decodable_string_batch, sos=False)
        decoder_outputs_array = to_categorical(
            decoder_outputs_array, num_classes=self.n_classes, dtype='float32'
        )
        
        return [natural_date_str_batch, decoder_inputs_array], decoder_outputs_array

    
"""
Model
"""
class natural_datetime_use:
    
    def define_train_model(self, use_path = "resources/muse"):
        # "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        self.embed = hub.KerasLayer("resources/muse", 
                                    input_shape=[],     # Expects a tensor of shape [batch_size] as input.
                                    trainable=True
                                    )
        self.LATENT_DIM = self.embed(['01 Apr 97']).shape[1]
        
        # https://github.com/tensorflow/hub/issues/648
        natural_date_str = keras.Input(shape=[], dtype=tf.string)
        encoding = self.embed(natural_date_str)

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, self.tokenizer.vocab_size))
        decoder_lstm = keras.layers.LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[encoding, encoding])
        decoder_dense = keras.layers.Dense(self.tokenizer.vocab_size, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.training_model = keras.Model([natural_date_str, decoder_inputs], decoder_outputs)
        self.training_model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        
        
    def define_decoder_model(self):
        # define resulting decoder
        decoder_inputs = self.training_model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.LATENT_DIM,), name="hiddenstate_input")
        decoder_state_input_c = keras.Input(shape=(self.LATENT_DIM,), name="cellstate_input")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.training_model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = self.training_model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        
        
    def train_on_synthetic_data(self, 
                                data_path = 'synthetic_datestrings.csv', 
                                batch_size = 64,
                                epochs=5
                               ):
        
        # train tokenizer
        df = pd.read_csv('synthetic_datestrings.csv', index_col = 0)
        self.tokenizer = decoder_tokenizer()
        self.tokenizer.fit(df.decodable_string)
        
        # define model
        self.define_train_model()
        
        # PREPARE TRAINING DATA
        training_generator = DataGenerator(df.iloc[:int(len(df)*0.8)], 
                                           tokenizer=self.tokenizer, 
                                           batch_size=batch_size)
        valid_generator = DataGenerator(df.iloc[int(len(df)*0.8):], 
                                        tokenizer=self.tokenizer, 
                                        batch_size=batch_size,
                                        noise=0)

        # train model
        self.training_model.fit_generator(
            generator=training_generator,
            validation_data=valid_generator,
            epochs=epochs,
        )
        
        self.define_decoder_model()

    def predict(self, natural_date_str):
        # get encodings
        encoding = self.embed([natural_date_str])

        # initialize decoder output
        decoder_input = np.zeros((1,1,self.tokenizer.vocab_size))
        sos_id = self.tokenizer.t.texts_to_sequences(['sos'])[0][0]
        eos_id = self.tokenizer.t.texts_to_sequences(['eos'])[0][0]
        decoder_input[:,:,sos_id]=1

        hidden_state = cell_state = encoding

        # Teacher forcing - feeding the target as the next input
        predicted_tokens, stop_condition = [], False
        while not stop_condition:

            # passing enc_output to the decoder
            # and get loss
            decoder_prediction, hidden_state, cell_state = self.decoder_model([decoder_input, 
                                                                              hidden_state, 
                                                                              cell_state]) 
            token_id = decoder_prediction.numpy().argmax()
            predicted_tokens.append(token_id)

            # Exit condition: either hit max length
            # or find stop character.
            if token_id == eos_id or len(predicted_tokens) > self.tokenizer.MAX_LEN:
                stop_condition = True

            # apply teacher forcing
            # prepare next (correct) token
            decoder_input = np.zeros((1,1,self.tokenizer.vocab_size))
            decoder_input[:,:,token_id]=1
        
        detokenized = self.tokenizer.detokenize([predicted_tokens])
        start , end = parse_decodable_str(detokenized)
        return start , end 

    def save(self, path="savedmodel"):
        self.tokenizer.save(f"{path}/tokenizer.json")
        tf.saved_model.save( keras.Sequential([keras.Input(shape=[], dtype=tf.string), self.embed]) , 
                            f"{path}/muse")
        self.decoder_model.save(f"{path}/decoder_model")
        
    def load(self, path="savedmodel"):
        self.tokenizer.load(f"{path}/tokenizer.json")
        self.embed = hub.KerasLayer(f"{path}/muse", 
                            input_shape=[],     # Expects a tensor of shape [batch_size] as input.
                            trainable=False
                            )
        self.decoder_model = tf.keras.models.load_model(f"{path}/decoder_model")
        