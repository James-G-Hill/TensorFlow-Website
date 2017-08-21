# Import gives access to all TensorFlow classes, methods and symbols
import os
import tensorflow as tf

# Hide CPU computation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# A computational graph is created from nodes
# A node takes 0+ tensors as inputs & produces a tensor as output

# Constants are 1 type of node
# Constants take no inputs & output an internally stored value
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # This has an implicit tf.float32
print(node1, node2)

# To evaluate the nodes we must run the computational graph with a 'session'
sess = tf.Session()
print(sess.run([node1, node2]))

# A new node can be created by a function such as 'add'
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# A graph can be parameterized with placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

# We can pass values to the graph with the 'feed_dict' argument to run method
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# Another operation can be created using an operation
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# Variables allow trainable parameters to be added to a graph
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Variable are not initialized when called
# They are initialized together through a special operation
init = tf.global_variables_initializer()
sess.run(init)

# We can pass an array into the placeholders
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# A model must be tested once created
# A placeholder is required for providing the comparison data
y = tf.placeholder(tf.float32)
# A loss function measures the difference between current model & provided data
# The standard loss mode for linear regression requires:
# Squaring the deltas between the current model & provided data
square_deltas = tf.square(linear_model - y)
# Then summing those squares
loss = tf.reduce_sum(square_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Variables can be changed using operations
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Optimizrs can slowly change variables to minimize a loss function
# The simplest optimizer is 'gradient descent'
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  # this resets values to incorrect defaults
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
