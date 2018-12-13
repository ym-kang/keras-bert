from keras_bert import get_base_dict, get_model, gen_batch_inputs
import keras
import tensorflow as tf
# A toy input example
sentence_pairs = [
    [['all', 'work', 'and', 'no', 'play'], ['makes', 'jack', 'a', 'dull', 'boy']],
    [['from', 'the', 'day', 'forth'], ['my', 'arm', 'changed']],
    [['and', 'a', 'voice', 'echoed'], ['power', 'give', 'me', 'more', 'power']],
]


# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word
inv_map = {v: k for k, v in token_dict.items()}

# Build & train the model
model = get_model(
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
)
model.summary()

def _generator():
    while True:
        yield gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=20,
            mask_rate=0.3,
            swap_sentence_rate=1.0,
        )

model.fit_generator(
    generator=_generator(),
    steps_per_epoch=200,
    epochs=3,
    validation_data=_generator(),
    validation_steps=100,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    ],
)


# Use the trained model
inputs, output_layer = get_model(  # `output_layer` is the last feature extraction layer (the last transformer)
    token_num=len(token_dict),
    head_num=5,
    transformer_num=12,
    embed_dim=25,
    feed_forward_dim=100,
    seq_len=20,
    pos_num=20,
    dropout_rate=0.05,
    training=False,  # The input layers and output layer will be returned if `training` is `False`
)

m2 = keras.models.Model(inputs=inputs,outputs=output_layer)
m2.compile(
    optimizer='adam',
    loss='mse',
    metrics={},
)
a = next(_generator())
r1 = m2.predict([a[0][0], a[0][2]])



import numpy as np 

input_txt = list(map(lambda x:inv_map[x],a[0][0][0]))
print("input: ")
print(input_txt)

prediction_txt = list(map(lambda x:inv_map[x[0]],a[1][0][0]))
print("prediction: ")
print(prediction_txt)