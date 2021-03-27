#requirements
#Tensorflow 2.3.0
#requirements for Tensorflow model 
#remember to restart before proceed

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip',
                 compression='zip', low_memory=False)

train_df, remaining = train_test_split(df, random_state=42, train_size=0.0075, stratify=df.target.values)
valid_df, _ = train_test_split(remaining, random_state=42, train_size=0.00075, stratify=remaining.target.values)

train_data = tf.data.Dataset.from_tensor_slices((train_df.question_text.values, train_df.target.values))
valid_data = tf.data.Dataset.from_tensor_slices((valid_df.question_text.values, valid_df.target.values))

label_list = [0, 1] # Label categories
max_seq_length = 128 # maximum length of (token) input sequences
train_batch_size = 32

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


# This provides a function to convert row to input features and label

def to_feature(text, label, label_list=label_list, max_seq_length=max_seq_length, tokenizer=tokenizer):
  example = classifier_data_lib.InputExample(guid = None,
                                            text_a = text.numpy(), 
                                            text_b = None, 
                                            label = label.numpy())
  feature = classifier_data_lib.convert_single_example(0, example, label_list,
                                    max_seq_length, tokenizer)
  
  return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)


# Wrap a Python func into a Tensorflow op for Eager Exec

def to_feature_map(text, label):
  input_ids, input_mask, segment_ids, label_id = tf.py_function(to_feature, inp=[text, label], 
                                Tout=[tf.int32, tf.int32, tf.int32, tf.int32])

  # py_func doesn't set the shape of the returned tensors.
  input_ids.set_shape([max_seq_length])
  input_mask.set_shape([max_seq_length])
  segment_ids.set_shape([max_seq_length])
  label_id.set_shape([])

  x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }
  return (x, label_id)


# train
train_data = (train_data.map(to_feature_map,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        #.cache()
                        .shuffle(1000)
                        .batch(32, drop_remainder=True)
                        .prefetch(tf.data.experimental.AUTOTUNE))

# valid
valid_data = (valid_data.map(to_feature_map,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
                        .batch(32, drop_remainder=True)
                        .prefetch(tf.data.experimental.AUTOTUNE)) 

# Building the model

def create_model():
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                      name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                  name="input_mask")
  input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                  name="input_type_ids")

  pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])

  drop = tf.keras.layers.Dropout(0.4)(pooled_output)
  output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(drop)

  model = tf.keras.Model(
    inputs={
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    },
    outputs=output)
  return model

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.BinaryCrossentropy(), #SparseCategoricalCrossentropy() if it is multi-class classiifier
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train model
epochs = 4
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=1)

model.evaluate(valid_data, verbose=1)

sample_example = ["Do you like flowers?",\
                  "Why all the bad guys wear a black hat?",\
                  "Why so many taxi drivers are so rude?",\
                  "May I have your email ID?",\
                  "How are you today?",\
                  "Why are you such a bad ass?"]
test_data = tf.data.Dataset.from_tensor_slices((sample_example, [0]*len(sample_example)))
test_data = (test_data.map(to_feature_map).batch(1))
preds = model.predict(test_data)
threshold = 5e-03#or 0.005    #0.5

['Toxic' if pred >=threshold else 'Sincere' for pred in preds]

#results
#1 "Do you like flowers?" 5.629e-04, Sincere
#2 "Why all the bad guys wear a black hat?" 3.348e-02, Toxic
#3 "Why so many taxi drivers are so rude?" 4.905e-03, Sincere
#4 "May I have your email ID?" 4.981e-05, Sincere
#5 "How are you today?" 2.120e-04, Sincere
#6 "Why are you such a bad ass?" 3.563e-01, Toxic

#FINAL RESULT
#All passed!!





