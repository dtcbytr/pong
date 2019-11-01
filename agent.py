import tensorflow as tf
from lstm import GRU
class Agent:
	def __init__(self, output_q_values_dims, output_mesg_dims):

		self.output_mesg_dims = output_mesg_dims
		self.output_q_values_dims = output_q_values_dims
		#Construct fully connected layers here and add trainable variables
		self.layers = []
		self.layers.append(tf.keras.layers.Dense(8, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(16, activation = tf.nn.relu))
		
		self.layers.append(GRU(16))
		self.layers.append(GRU(16))
		self.layers.append(GRU(8))

		self.gru_layers = 3

		self.q_values = tf.keras.layers.Dense(output_q_values_dims)
		self.message = tf.keras.layers.Dense(output_mesg_dims, activation = tf.nn.tanh)

		self.FLAG = True

	def forward_prop(self, view_input, mesg_input):	#returns q values, message_vector
		output_tensor = tf.concat([view_input, mesg_input], axis = 1)
		for layer in self.layers:
			output_tensor = layer(output_tensor)
		if self.FLAG:
			self.FLAG = False
			self.trainable_variables = self.q_values.trainable_variables + self.message.trainable_variables

			for layer in self.layers:
				self.trainable_variables += layer.trainable_variables

		return self.q_values(output_tensor), self.message(output_tensor)

	def get_weights(self):
		return_value = []
		for layer in self.layers + [self.q_values, self.message]:
			return_value.append(layer.get_weights())
		return return_value

	def set_weights(self, weights):
		layer_list = self.layers + [self.q_values, self.message]
		for i, parameter in enumerate(weights):
			layer_list[i].set_weights(parameter)

	def reset(self):
		for layer in self.layers[-self.gru_layers: ]:
			layer.reset()