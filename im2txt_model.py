from keras.layers import Dense, Embedding, LSTM, concatenate, Input, Lambda, TimeDistributed, AveragePooling2D, \
    Reshape
from keras.models import Model
from cnn_model import create_nas_model, load_nas_model
from keras.backend import expand_dims


def create_training_model(dictionary_size=9568, hidden_size=512):
    input_cnn = Input(shape=(224, 224, 3), dtype='float32')

    nas_model = load_nas_model()
    nas_model.trainable = False

    feature_cnn = nas_model(input_cnn)
    avg_bool = AveragePooling2D(pool_size=(int(feature_cnn.shape[1]), int(feature_cnn.shape[2])))(feature_cnn)
    feature_flat = Reshape((-1,))(avg_bool)
    embed_features = Dense(hidden_size, activation='relu')(feature_flat)

    input_caption = Input(shape=(None,), dtype='int32')

    embed_caption = Embedding(output_dim=hidden_size, input_dim=dictionary_size)(
        input_caption)

    expanded_features = Lambda(lambda x: expand_dims(x, axis=1))(embed_features)

    conncat_layer = concatenate([expanded_features, embed_caption], axis=1)

    lstm_out = LSTM(hidden_size, return_sequences=True, stateful=False)(conncat_layer)

    output = TimeDistributed(Dense(dictionary_size, activation='softmax'))(lstm_out)

    model = Model(inputs=[input_cnn, input_caption], outputs=[output])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    return model


def create_prediction_cnn(hidden_size=512, weighted_model=None):
    input_cnn = Input(batch_shape=(1, 224, 224, 3), dtype='float32', name="input_1")

    nas_model = load_nas_model()
    nas_model.trainable = False

    feature_cnn = nas_model(input_cnn)
    avg_bool = AveragePooling2D(pool_size=(int(feature_cnn.shape[1]), int(feature_cnn.shape[2])),
                                name="average_pooling2d_1")(feature_cnn)
    feature_flat = Reshape((-1,), name="reshape_1")(avg_bool)
    embed_features = Dense(hidden_size, activation='relu', name="dense_1")(feature_flat)
    expanded_features = Lambda(lambda x: expand_dims(x, axis=1), name="lambda_1")(embed_features)

    lstm_out, state_h, state_c = LSTM(hidden_size, return_sequences=True, return_state=True, stateful=True,
                                      name='lstm_1')(expanded_features)
    model = Model(inputs=[input_cnn], outputs=[state_h, state_c])
    if weighted_model is not None:
        for weighted_layer in weighted_model.layers:
            try:
                layer = model.get_layer(weighted_layer.name)
                layer.set_weights(weighted_layer.get_weights())
            except:
                continue
    return model


def create_prediction_lstm(dictionary_size=9568, hidden_size=512, weighted_model=None, batch_size=3):
    input_caption = Input(batch_shape=(batch_size, None), dtype='int32', name='input_3')

    embed_caption = Embedding(output_dim=hidden_size, input_dim=dictionary_size, name='embedding_1')(
        input_caption)

    input_state_h = Input(shape=(512,), dtype='float32', name='input_state_h')
    input_state_c = Input(shape=(512,), dtype='float32', name='input_state_c')
    lstm_out, state_h, state_c = LSTM(hidden_size, return_sequences=True, return_state=True, stateful=True,
                                      name='lstm_1')(embed_caption, initial_state=[input_state_h, input_state_c])

    output = TimeDistributed(Dense(dictionary_size, activation='softmax'), name='time_distributed_1')(lstm_out)

    model = Model(inputs=[input_caption, input_state_h, input_state_c],
                  outputs=[output, state_h, state_c])

    if weighted_model is not None:
        for weighted_layer in weighted_model.layers:
            try:
                layer = model.get_layer(weighted_layer.name)
                layer.set_weights(weighted_layer.get_weights())
            except:
                continue
    return model
