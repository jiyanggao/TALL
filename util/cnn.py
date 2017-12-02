from __future__ import division  #, print_function

import tensorflow as tf

def conv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
               bias_term=True, weights_initializer=None, biases_initializer=None):
    # input has shape [batch, in_height, in_width, in_channels]
    input_dim = bottom.get_shape().as_list()[-1]

    # weights and biases variables
    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            if bias_term and biases_initializer is None:
              biases_initializer = tf.constant_initializer(0.)
            print "input_dim"+str(input_dim)
            # filter has shape [filter_height, filter_width, in_channels, out_channels]
            weights = tf.get_variable("weights",
                [kernel_size, kernel_size, input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)
            print str(weights.name)+" initialized as random or retrieved from graph"
            if bias_term:
                print biases.name+" initialized as random or retrieved from graph"

        else:
            weights = tf.get_variable("weights",
                shape=None,
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", shape=None,
                    initializer=biases_initializer) 

            print weights.name+" initialized from pre-trained parameters or retrieved from graph"
            if bias_term:
                print biases.name+" initialized from pre-trained parameters or retrieved from graph"


    conv = tf.nn.conv2d(bottom, filter=weights,
        strides=[1, stride, stride, 1], padding=padding)
    if bias_term:
        conv = tf.nn.bias_add(conv, biases)
    return conv

def conv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                    bias_term=True, weights_initializer=None, biases_initializer=None):
    conv = conv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                      bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(conv)
    return relu

def deconv_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                 bias_term=True, weights_initializer=None, biases_initializer=None):
    # input_shape is [batch, in_height, in_width, in_channels]
    input_shape = bottom.get_shape().as_list()
    batch_size, input_height, input_width, input_dim = input_shape
    output_shape = [batch_size, input_height*stride, input_width*stride, output_dim]

    # weights and biases variables
    with tf.variable_scope(name):
        # initialize the variables
        if weights_initializer is None:
            weights_initializer = tf.random_normal_initializer()
        if bias_term and biases_initializer is None:
            biases_initializer = tf.constant_initializer(0.)

        # filter has shape [filter_height, filter_width, out_channels, in_channels]
        weights = tf.get_variable("weights",
            [kernel_size, kernel_size, output_dim, input_dim],
            initializer=weights_initializer)
        if bias_term:
            biases = tf.get_variable("biases", output_dim,
                initializer=biases_initializer)

    deconv = tf.nn.conv2d_transpose(bottom, filter=weights,
        output_shape=output_shape, strides=[1, stride, stride, 1],
        padding=padding)
    if bias_term:
        deconv = tf.nn.bias_add(deconv, biases)
    return deconv

def deconv_relu_layer(name, bottom, kernel_size, stride, output_dim, padding='SAME',
                      bias_term=True, weights_initializer=None, biases_initializer=None):
    deconv = deconv_layer(name, bottom, kernel_size, stride, output_dim, padding,
                          bias_term, weights_initializer, biases_initializer)
    relu = tf.nn.relu(deconv)
    return relu

def pooling_layer(name, bottom, kernel_size, stride):
    pool = tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, stride, stride, 1], padding='SAME', name=name)
    return pool

def fc_layer(name, bottom, output_dim, bias_term=True, weights_initializer=None,
             biases_initializer=None):
    # flatten bottom input
    # input has shape [batch, in_height, in_width, in_channels]
    shape = bottom.get_shape().as_list()
    input_dim = 1
    for d in shape[1:]:
        input_dim *= d
    flat_bottom = tf.reshape(bottom, [-1, input_dim])
    
    # weights and biases variables
    with tf.variable_scope(name):
        if weights_initializer is None and biases_initializer is None:
            # initialize the variables
            if weights_initializer is None:
                weights_initializer = tf.random_normal_initializer()
            if bias_term and biases_initializer is None:
                biases_initializer = tf.constant_initializer(0.)

            # weights has shape [input_dim, output_dim]
            weights = tf.get_variable("weights", [input_dim, output_dim],
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", output_dim,
                    initializer=biases_initializer)

            print weights.name+" initialized as random or retrieved from graph"
            if bias_term:
                print biases.name+" initialized as random or retrieved from graph"
        else:
            weights = tf.get_variable("weights", shape=None,
                initializer=weights_initializer)
            if bias_term:
                biases = tf.get_variable("biases", shape=None,
                    initializer=biases_initializer)

            print weights.name+" initialized from pre-trained parameters or retrieved from graph"
            if bias_term:
                print biases.name+" initialized from pre-trained parameters or retrieved from graph"

    if bias_term:
        fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
    else:
        fc = tf.matmul(flat_bottom, weights)
    return fc

def fc_relu_layer(name, bottom, output_dim, bias_term=True,
                  weights_initializer=None, biases_initializer=None):
    fc = fc_layer(name, bottom, output_dim, bias_term, weights_initializer,
                  biases_initializer)
    relu = tf.nn.relu(fc)
    return relu

def softmax_loss_layer(name, score_bottom, label_bottom):
    """
    Calculates cumulative Softmax Cross Entropy Loss along the last dimension
    *This function does not divide the loss by batch size*

    Once tensorflow has SparseCrossEntropy function, this one will be replaced
    """
    # Check shape
    score_shape = score_bottom.get_shape().as_list()
    label_shape = label_bottom.get_shape().as_list()
    assert len(score_shape) == len(label_shape) + 1
    assert score_shape[:-1] == label_shape

    # Compute the outer dimensions dimensions in label
    inner_dim = score_shape[-1]
    outer_dim = 1
    for d in label_shape: outer_dim *= d

    # flatten score and label
    flat_score = tf.reshape(score_bottom, [outer_dim, inner_dim])
    flat_label = tf.reshape(label_bottom, [outer_dim, 1])

    # Reshape the labels into a dense Tensor of
    # shape [batch_size, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    dense_labels = tf.sparse_to_dense(concated, [FLAGS.batch_size, NUM_CLASSES],
        1.0, 0.0)
