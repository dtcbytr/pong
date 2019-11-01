import tensorflow as tf

class Mediator:
	def __init__(self, output_dims):
		self.output_dims = output_dims
		# Construct the layers here, with the last layer having sigmoid activation and add trainable_variables
		self.layers = []
		self.layers.append(tf.keras.layers.Dense(units = 8, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = 16, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = 16, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = output_dims, activation = tf.nn.tanh))

		self.FLAG = True

	def forward_prop(self, agent_inputs):
		output_tensor = tf.concat(agent_inputs, axis = 1)
		for layer in self.layers:
			output_tensor = layer(output_tensor)

		if self.FLAG:
			self.FLAG = False
			self.trainable_variables = []
			for layer in self.layers:
				self.trainable_variables += layer.trainable_variables

		return output_tensor

	def get_weights(self):
		res = []
		for layer in self.layers:
			res.append(layer.get_weights())
		return res
	def set_weights(self, weights):
		for i, weight in enumerate(weights):
			self.layers[i].set_weights(weight)
