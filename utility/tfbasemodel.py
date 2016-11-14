import tensorflow as tf
import numpy as np
import logging
import sys



    
class TFModel(object):
    def __init__(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(sys.stdout))
        root.setLevel(logging.DEBUG)
        return
    def get_next_batch(self, x, y, batch_size):
        """
        Shuffle a dataset and randomly fetch next batch
        """
        _positions = np.random.choice(x.shape[0], size=batch_size, replace=False)
        batch_data = x[_positions]
        batch_labels = y[_positions]
        return batch_data, batch_labels
    def get_input(self):
        pass
    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        #this applies to both cnn and nn
        input_dims = shape[:-1]
        input_num = np.array(input_dims).prod()
        
        weight_scale = np.sqrt(2.0/input_num)
#         weight_scale = 1e-2
        initial = (weight_scale * np.random.randn(*shape)).astype(np.float32)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial)
    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
        return
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    def max_pool_2x2(self, layer_name, x):
        with tf.name_scope(layer_name):
            post_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
            return post_pool
    def cnn_layer(self,layer_name, input_tensor,  conv_fitler=[5,5,10], act=tf.nn.relu, dropout=True, batch_norm = True):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        in_channels = input_tensor.get_shape()[-1].value
        filter_height, filter_width, output_channels = conv_fitler
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                # weight dimension, filter_height, filter_width,in_channels, output_channels
                weights = self.weight_variable([filter_height, filter_width,in_channels, output_channels])
                self.variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_channels])
                self.variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                out = self.conv2d(input_tensor, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', out)               
            
            if batch_norm:
                out = self.batch_norm(out, self.phase_train_placeholder)
                
            if act is not None:
                out = act(out, 'activation')
                tf.histogram_summary(layer_name + '/activations', out)
                
            if dropout:
                out = self.dropout_layer(out)
        return out
    def nn_layer(self,layer_name, input_tensor, output_dim,  act=tf.nn.relu, dropout=True, batch_norm = True):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        input_dim = [dim.value for dim in input_tensor.get_shape().dims][1:]
        input_dim = np.array(input_dim).prod()
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights, layer_name + '/weights')
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases, layer_name + '/biases')
            with tf.name_scope('Wx_plus_b'):
                input_tensor_flat = tf.reshape(input_tensor, [-1, input_dim])
                out = tf.matmul(input_tensor_flat, weights) + biases
                tf.histogram_summary(layer_name + '/pre_activations', out)               
            
            if batch_norm:
                out = self.batch_norm(out, self.phase_train_placeholder)
                
            if act is not None:
                out = act(out, 'activation')
                tf.histogram_summary(layer_name + '/activations', out)
                
            if dropout:
                out = self.dropout_layer(out)
        return out
    
    def dropout_layer(self, to_be_dropped_layer):
        layer_id = int(to_be_dropped_layer.name.split('/')[0][-1])
        with tf.name_scope('dropout' + str(layer_id)):
            dropped = tf.nn.dropout(to_be_dropped_layer, self.keep_prob_placeholder)
            return dropped
    def batch_norm(self, input_tensor,phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            input_tensor:           Tensor, 4D BHWD input maps
            feature_depth:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        inputs_shape = input_tensor.get_shape()
        feature_depth = inputs_shape[-1].value
        reduce_axis = list(range(len(inputs_shape) - 1))
        layer_id = int(input_tensor.name.split('/')[0][-1])
        with tf.variable_scope('bn'+  str(layer_id)):
            beta = tf.Variable(tf.constant(0.0, shape=[feature_depth]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[feature_depth]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(input_tensor, reduce_axis, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)
    
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
    
            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(input_tensor, mean, var, beta, gamma, 1e-3)
        return normed

    def add_inference_node(self):
        pass
    def add_loss_node(self):
        pass
    def add_optimizer_node(self):
        pass
    def add_evalmetrics_node(self):
        pass
    def add_visualize_node(self):
        pass
    def __build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.get_input()
            self.add_inference_node()
            self.add_loss_node()
            self.add_optimizer_node()
            self.add_evalmetrics_node()
            self.add_visualize_node()
        return
    def run_graph(self):
        return
    def clear_prev_summary(self):
        if tf.gfile.Exists(self.summaries_dir):
            tf.gfile.DeleteRecursively(self.summaries_dir)
        tf.gfile.MakeDirs(self.summaries_dir)
        return
    def run(self):
        self.clear_prev_summary()
        self.__build_graph()
        self.run_graph()
        return