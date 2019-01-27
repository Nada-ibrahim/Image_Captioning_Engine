from keras.layers import Dense, Embedding, LSTM, concatenate, Input, Lambda, TimeDistributed, AveragePooling2D, \
    Reshape
from keras.models import Model
from cnn_model import create_nas_model
from keras.backend import expand_dims
from keras.utils import plot_model


def create_model(dictionary_size, max_seq_length, hidden_size=512, stateful = False):
    if stateful:
        input_cnn = Input(batch_shape=(1,224, 224, 3), dtype='float32')
    else:
        input_cnn = Input(shape=(224, 224, 3), dtype='float32')

    nas_model = create_nas_model()
    nas_model.trainable = False

    feature_cnn = nas_model(input_cnn)
    avg_bool = AveragePooling2D(pool_size=(int(feature_cnn.shape[1]), int(feature_cnn.shape[2])))(feature_cnn)
    feature_flat = Reshape((-1,))(avg_bool)
    embed_features = Dense(hidden_size, activation='relu')(feature_flat)
    expanded_features = Lambda(lambda x: expand_dims(x, axis=1))(embed_features)

    if stateful:
        input_caption = Input(batch_shape=(1,None), dtype='int32')
    else:
        input_caption = Input(shape=(None,), dtype='int32')

    embed_caption = Embedding(output_dim=hidden_size, input_dim=dictionary_size)(
        input_caption)

    conncat_layer = concatenate([expanded_features, embed_caption], axis=1)

    lstm_out = LSTM(hidden_size, return_sequences=True, stateful=stateful)(
        conncat_layer)

    output = TimeDistributed(Dense(dictionary_size, activation='softmax'))(lstm_out)

    model = Model(inputs=[input_cnn, input_caption], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    for i in range(len(model.layers)):
        print(model.layers[i])
    return model

create_model(9000, 52)