from pong import Pong
import cv2 as cv
from agent import Agent
import config
from mediator import Mediator
import numpy as np
import tensorflow as tf

def sample_action(alpha, q_values):	 # alpha is the probability with which the agent takes a random action
	if(np.random.uniform() < alpha):
		return np.random.randint(q_values.shape[-1])
	else:
		return tf.argmax(q_values, axis = 1).numpy().item()

def run(agents, mediator, pong):
	pong.reset()
	for agent in agents:
		agent.reset()

	message_broadcast = tf.random.normal(shape = [1, config.AGENT_MESG_INPUT_DIMS])
	rewards = 0.0
	done = False
	time = 0
	total_reward_l, total_reward_r = 0.0, 0.0
	while not done:
		state_a = [pong.get_left_observation(config.VIEW_L)]
		state_b = [pong.get_right_observation(config.VIEW_R)]
		q_values_a, message_a = agents[0].forward_prop(state_a, message_broadcast)
		q_values_b, message_b = agents[1].forward_prop(state_b, message_broadcast)
		# now run through mediator
		message_list = [message_a, message_b]
		message_broadcast = mediator.forward_prop(message_list)
		# now get the next state, run this info through the agents and remove the preivous to previous screen from states vairable
		action_a = tf.argmax(q_values_a, axis = 1)[0]
		action_b = tf.argmax(q_values_b, axis = 1)[0]
		pong.render()
		reward_l, reward_r, done = pong.step(action_a.numpy(), action_b.numpy())
		total_reward_l += reward_l
		total_reward_r += reward_r
		time += 1
	return total_reward_l, total_reward_r, time

def train_episode(agents, target_agents, mediator, target_mediator, pong):
	pong.reset()
	for agent in agents + target_agents:
		agent.reset()
	state_a = [pong.get_left_observation(config.VIEW_L)]
	state_b = [pong.get_right_observation(config.VIEW_R)]

	message_broadcast = tf.random.normal(shape = [1, config.AGENT_MESG_INPUT_DIMS])
	target_message_broadcast = tf.random.normal(shape = [1, config.AGENT_MESG_INPUT_DIMS])
	_, target_message_a = target_agents[0].forward_prop(state_a, target_message_broadcast)
	_, target_message_b = target_agents[1].forward_prop(state_b, target_message_broadcast)
	target_message_broadcast = target_mediator.forward_prop([target_message_a, target_message_b])
	rewards = 0.0
	total_reward_l, total_reward_r = 0.0, 0.0
	done = False
	time = 0
	loss = tf.constant(0.0)
	while not done:
		q_values_a, message_a = agents[0].forward_prop(state_a, message_broadcast)
		q_values_b, message_b = agents[1].forward_prop(state_b, message_broadcast)
		# now run through mediator
		message_list = [message_a, message_b]
		message_broadcast = mediator.forward_prop(message_list)
		# now get the next state, run this info through the agents and remove the preivous to previous screen from states vairable
		action_a = sample_action(config.EXPLORE_PROB, q_values_a)
		action_b = sample_action(config.EXPLORE_PROB, q_values_b)
		pong.render()
		reward_l, reward_r, done = pong.step(action_a, action_b)
		total_reward_l += reward_l
		total_reward_r += reward_r
		rewards += reward_l + reward_r
		if done:
			loss += tf.square(reward_l - q_values_a[0, action_a]) + tf.square(reward_r - q_values_b[0, action_b])
		else:
			state_a = [pong.get_left_observation(config.VIEW_L)]
			state_b = [pong.get_right_observation(config.VIEW_R)]

			target_q_values_a, target_message_a = target_agents[0].forward_prop(state_a, target_message_broadcast)
			target_q_values_b, target_message_b = target_agents[1].forward_prop(state_b, target_message_broadcast)
			target_message_broadcast = target_mediator.forward_prop([target_message_a, target_message_b])
			loss += tf.square(reward_l + config.DISCOUNT_FACTOR * tf.reduce_max(target_q_values_a) - q_values_a[0, action_a]) + tf.square(reward_r + config.DISCOUNT_FACTOR * tf.reduce_max(target_q_values_b) - q_values_b[0, action_b])
		time += 1

	return total_reward_l, total_reward_r, time, loss

def update_target_network(agents, target_agents):
	for target_agent, agent in zip(target_agents, agents):
		target_agent.set_weights(agent.get_weights())

def main():
	description = 'no_par_shar-random_init-vector_state'
	pong = Pong(description)
	
	agents = [Agent(3, config.AGENT_MESG_OUTPUT_DIMS), Agent(3, config.AGENT_MESG_OUTPUT_DIMS)]
	target_agents = [Agent(3, config.AGENT_MESG_OUTPUT_DIMS), Agent(3, config.AGENT_MESG_OUTPUT_DIMS)]

	mediator = Mediator(config.AGENT_MESG_INPUT_DIMS)
	target_mediator = Mediator(config.AGENT_MESG_INPUT_DIMS)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate = config.LEARNING_RATE)

	# to initialize, we run a random episode
	run(agents, mediator, pong)
	run(target_agents, target_mediator, pong)
	update_target_network(agents + [mediator], target_agents + [target_mediator])
	episodes = 0
	while True:
		with tf.GradientTape() as tape:
			reward_l, reward_r, time, loss = train_episode(agents, target_agents, mediator, target_mediator, pong)

		trainable_variables = agents[0].trainable_variables + agents[1].trainable_variables + mediator.trainable_variables
		grads = tape.gradient(loss, trainable_variables)
		optimizer.apply_gradients(zip(grads, trainable_variables))
		episodes += 1

		print('Train Episode ({}) {:7d}:	Reward_l: {:7.4f}	Reward_r: {:7.4f}	Timesteps: {:4d}'.format(description, episodes, reward_l, reward_r, time))
		if episodes % config.EPISODES_TO_TRAIN == 0:
			reward_l, reward_r, time = run(agents, mediator, pong)
			print('Test Episode {:7d}:	Reward_l: {:7.4f}	Reward_r: {:7.4f}	Timesteps: {:4d}'.format(episodes, reward_l, reward_r, time))
			update_target_network(agents + [mediator], target_agents + [target_mediator])


if __name__ == '__main__':
	main()