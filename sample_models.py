from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM,
     MaxPooling1D, Add)
from keras import regularizers

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=1, name='rnn')(input_data)
    simp_rnn = Dropout(rate=0.1)(simp_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    bn_rnn = Dropout(rate=0.1)(bn_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    bn_cnn = Dropout(rate=0.1)(bn_cnn)
    # Add a recurrent layer
    simp_rnn = GRU(units, 
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_simp_rnn')(simp_rnn)
    bn_rnn = Dropout(rate=0.1)(bn_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    deep_rnn = input_data
    for i in range(recur_layers):
        deep_rnn = GRU(units, return_sequences=True, 
                       implementation=2)(deep_rnn)
        deep_rnn = BatchNormalization()(deep_rnn)
        deep_rnn = Dropout(rate=0.1)(deep_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(deep_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True, 
               implementation=2, name='rnn'))(input_data)
    bidir_rnn = Dropout(rate=0.1)(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, cnn_mode, cnn_dim, cnn_units, cnn_layers,
           rnn_mode, rnn_dim, rnn_layers, imp_mode, dense_dim,
           dense_layers, l2rate, droprate, output_dim=29):
    """ Build a deep network for speech
        Args:
            cnn_mode:  cnn/dialat/tcn
            rnn_mode:  gru/bid
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    assert imp_mode is 1 or imp_mode is 2, \
        "Value error of implementation mode, should be 1 or 2"
    regu = regularizers.l2(l2rate) if l2rate else None
    
    def conv_layer(net, dim_num, ksize, units_num, layer_num):
        """ Build a Deep CNN network
        """
        for i in range(units_num * layer_num):
            kernel_size = ksize if i else 11
            strides = 1 if i else 2
            padding = 'same' if i else 'valid'
            net = Conv1D(filters=dim_num,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    activation='relu',
                    kernel_regularizer=regu)(net)
            net = BatchNormalization()(net)
            if droprate:
                net = Dropout(rate=droprate)(net)
        return net

    def dialat_layer(net, dim_num, ksize, units_num, layer_num):
        """ Build a Dialation CNN deep network
        """
        for i in range(layer_num):
            for j in range(units_num):
                net = Conv1D(filters=dim_num,
                        kernel_size=ksize,
                        strides=1,
                        padding='same',
                        dilation_rate=2**j,
                        activation='relu',
                        kernel_regularizer=regu)(net)
                net = BatchNormalization()(net)
                if droprate:
                    net = Dropout(rate=droprate)(net)
        return net

    def tcn_layer(net, dim_num, ksize, units_num, layer_num):
        """ Build a TCN deep network
        """
        for i in range(layer_num):
            shortcut = Conv1D(filters=dim_num,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        activation='relu',
                        kernel_regularizer=regu)(net)
            for j in range(units_num):
                net = Conv1D(filters=dim_num,
                        kernel_size=ksize,
                        strides=1,
                        padding='same',
                        dilation_rate=2**j,
                        activation='relu',
                        kernel_regularizer=regu)(net)
                net = BatchNormalization()(net)
                if droprate:
                    net = Dropout(rate=droprate)(net)
            net = Add()([net, shortcut])
        return net

    def rnn_layer(net, dim_num, layer_num, use_bid):
        for i in range(layer_num):
            if use_bid:
                net = Bidirectional(GRU(dim_num,
                                return_sequences=True,
                                implementation=imp_mode,
                                kernel_regularizer=regu))(net)
            else:
                net = GRU(dim_num,
                       return_sequences=True,
                       implementation=imp_mode,
                       kernel_regularizer=regu)(net)

            net = BatchNormalization()(net)
            if droprate:
                net = Dropout(rate=droprate)(net)
        return net
    
    def dense_layer(net, dim_num, layer_num):
        assert layer_num > 0
        for i in range(layer_num):
            is_end_layer = i == layer_num-1
            units = output_dim if is_end_layer else dim_num
            net = TimeDistributed(Dense(units, kernel_regularizer=regu))(net)
            if not is_end_layer:
                net = BatchNormalization()(net)
                if droprate:
                    net = Dropout(rate=droprate)(net)
        return net

    # Take the border='same' and stride=1 to makes the output length
    # only determined by layer 'conv1d_input'
    if cnn_mode == 'cnn':
        net = conv_layer(input_data, cnn_dim, 3, cnn_units, cnn_layers)
    elif cnn_mode == 'dialat':
        net = dialat_layer(input_data, cnn_dim, 3, cnn_units, cnn_layers)
    elif cnn_mode == 'tcn':
        net = tcn_layer(input_data, cnn_dim, 3, cnn_units, cnn_layers)
    else:
        net = input_data

    if rnn_mode == 'gru':
        net = rnn_layer(net, rnn_dim, rnn_layers, use_bid=False)
    elif rnn_mode == 'bid':
        net = rnn_layer(net, rnn_dim, rnn_layers, use_bid=True)

    net = dense_layer(net, dense_dim, dense_layers)

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(net)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    if cnn_mode != 'cnn':
        model.output_length = lambda x: x
    else:
        model.output_length = lambda x: cnn_output_length(
            x, filter_size=11, border_mode='valid', stride=2)
    print(model.summary())
    return model