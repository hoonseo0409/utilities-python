import tensorflow as tf

def is_in(item, container):
    pass

def tensor_for_loop(body, body_vars, loop_kwargs = None, idx_range = (0, 5)):
    raise Exception(NotImplementedError)
    if not isinstance(idx_range, (tuple, list)):
        idx_range = (0, idx_range)
    if loop_kwargs is None:
        loop_kwargs = {}
    idx = tf.constant(idx_range[0])
    def local_body(idx, *body_vars):
        return idx + 1, body(idx, *body_vars)
    return tf.while_loop(lambda idx, *body_vars: tf.less(idx, idx_range[1]), local_body, [idx, *body_vars], **loop_kwargs)

class ForLoopLayer(tf.keras.layers.Layer):
  def __init__(self):
    """
    Example: https://stackoverflow.com/questions/71635459/how-to-use-keras-symbolic-inputs-with-tf-while-loop

    inputs = tf.keras.layers.Input(shape= (None, ), batch_size= 1, name= "timesteps", dtype= tf.int32)
    cl = CustomLayer()
    outputs = cl(inputs)
    model = tf.keras.Model(inputs, outputs)
    random_data = tf.random.uniform((1, 7), dtype=tf.int32, maxval=50)
    print(model(random_data))

    """
    super(ForLoopLayer, self).__init__()
    raise Exception(NotImplementedError)
              
  def call(self, inputs, body, funct_end, funct_start= lambda inputs: 0, funct_step= lambda inputs: 1, dtype= tf.float32):

    # input_shape = tf.shape(inputs)
    # end = input_shape[-1]
    # array = tf.ones((input_shape[-1],))
    # start = tf.constant(0)
    # inputs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    end = funct_end(inputs)
    start = funct_start(inputs)
    step = funct_step(inputs)

    it = tf.constant(start)
    outputs = tf.TensorArray(dtype= dtype, size=0, dynamic_size=True)

    def cond(it, outputs, inputs):
        return it < end

    def local_body(it, outputs, inputs):
        # inputs = inputs.write(it, tf.gather(array, it))
        outputs = outputs.write(it, body(it, outputs, inputs))
        return it + step, outputs, inputs

    _, outputs, _ = tf.while_loop(cond, local_body, loop_vars=[it, outputs, inputs])
    return outputs

def element_is_in_tensor(tensor, elem):
    equalities = tf.math.equal(tensor, elem)
    return tf.math.reduce_any(equalities)

if __name__ == "__main__":
    if False:
        myinput = tf.keras.layers.Input(shape= (None, ), batch_size= 1, name= "timesteps", dtype= tf.int32)
        inputs = [tf.ones((tf.shape(myinput)[-1],)), myinput]
        cl = ForLoopLayer()
        def mybody(it, outputs, inputs):
            return tf.gather(inputs[0], it)
        outputs = cl(inputs, mybody, lambda inputs: tf.shape(inputs[1])[-1])
        model = tf.keras.Model(inputs, outputs)
        random_data = tf.random.uniform((1, 7), dtype=tf.int32, maxval=50)
        print(model(random_data))