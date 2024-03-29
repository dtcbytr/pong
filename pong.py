import numpy as np
import cv2 as cv
from math import ceil, floor
import math

def check(x, x_max):
	return x >= 0 and x < x_max

class Ball:
	def __init__(self, x_init, y_init, vx_init, vy_init, x_max, y_max):
		self.x = x_init
		self.y = y_init
		self.speed = (vx_init ** 2 + vy_init ** 2) ** 0.5
		self.vx = vx_init / self.speed
		self.vy = vy_init / self.speed
		self.x_max = x_max
		self.y_max = y_max
		self.x_init = x_init
		self.y_init = y_init
		self.vx_init = self.vx
		self.vy_init = self.vy

	def move(self):
		cx, cy = (-1, -1)
		if(check(self.x + self.vx * self.speed, self.x_max)):
			self.x += self.vx * self.speed
			cx = 1
		if(check(self.y + self.vy * self.speed, self.y_max)):
			self.y += self.vy * self.speed
			cy = 1
		self.vx *= cx
		self.vy *= cy
		if(cx == -1 or cy == -1):
			self.move()

	def reset(self):
		self.x = self.x_init
		self.y = self.y_init
		speed = (self.vx_init ** 2 + self.vy_init ** 2) ** 0.5
		self.vx = np.random.uniform(-speed * 3.0 / 5.0, speed * 3.0 / 5.0)
		self.vy = ((speed ** 2 - self.vx ** 2) ** 0.5) * (1.0 if np.random.uniform() < 0.5 else -1.0)
	def reflect(self):
		self.vy = -self.vy


class Paddle:
	def __init__(self, x, length, x_max):
		self.x = x
		self.x_init = x
		self.x_max = x_max
		self.length = length

	def reset(self):
		self.x = self.x_init

	def move(self, dx):
		if(check(self.x + dx, self.x_max) and check(self.x + dx + self.length - 1, self.x_max)):
			self.x += dx

class Pong:
	def __init__(self, window_name, x_init = 50, y_init = 50, vx_init = 0.3, vy_init = 1.0, x_paddle_left = 45, x_paddle_right = 45):
		self.window_name = window_name
		cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
		cv.resizeWindow(self.window_name, 100, 100)

		self.window_x = 100
		self.window_y = 100
		self.paddle_length = 15
		self.paddle_width = 1
		self.left_paddle_position = 5
		self.right_paddle_position = self.window_y - self.left_paddle_position - 1
		self.ball = Ball(x_init, y_init, vx_init, vy_init, self.window_x, self.window_y)
		self.paddle_left = Paddle(x_paddle_left, self.paddle_length, self.window_x)
		self.paddle_right = Paddle(x_paddle_right, self.paddle_length, self.window_x)
		self.x_init, self.y_init = (x_init, y_init)
		self.vx_init, self.vy_init = (vx_init, vy_init)
		self.x_paddle_left, self.x_paddle_right = x_paddle_left, x_paddle_right

	def reset(self):
		self.ball.reset()
		self.paddle_left.reset()
		self.paddle_right.reset()

	def get_screen(self):
		screen = np.zeros([self.window_x, self.window_y])
		cv.line(screen, pt1 = (self.left_paddle_position, self.paddle_left.x), pt2 = (self.left_paddle_position, self.paddle_length + self.paddle_left.x), color = 1.0, lineType = 0, thickness = self.paddle_width)
		cv.line(screen, pt1 = (self.right_paddle_position, self.paddle_right.x), pt2 = (self.right_paddle_position, self.paddle_length + self.paddle_right.x), color = 1.0, lineType = 0, thickness = self.paddle_width)
		if(self.ball.x - floor(self.ball.x) > 0.5):
			c_x = ceil(self.ball.x)
		else:
			c_x = floor(self.ball.x) - 1
		if(self.ball.y - floor(self.ball.y) > 0.5):
			c_y = ceil(self.ball.y)
		else:
			c_y = floor(self.ball.y) - 1
		x, y = (floor(self.ball.x), floor(self.ball.y))
		points = [(c_y, c_x), (c_y, x), (y, c_x), (y, x)]
		# dist = [abs(x - self.ball.x) + abs(y - self.ball.y) for x, y in points]
		# sorted_list = sorted(zip(dist, points))
		# cv.circle(screen, (int(self.ball.y), int(self.ball.x)), 0, lineType = 4, color = 255)
		intensity_controls = [0.6, 0.6, 0.6, 1.0]
		for i in range(4):
			# cv.circle(screen, points[i], 0, lineType = 8, color = intensity_controls[i])
			if check(points[i][1], self.window_y) and check(points[i][0], self.window_x):
				screen[points[i][1], points[i][0]] = intensity_controls[i]

		return screen

	def get_state(self):	# returns a numpy array of shape [4]
		return np.asarray([self.ball.x, self.ball.y, self.paddle_left.x, self.paddle_right.x])

	def get_left_observation(self, view_l):
		visible = [1.0, self.ball.x, self.ball.y] if self.ball.y < view_l else [0.0, 0.0, 0.0]
		return np.asarray(visible + [self.paddle_left.x])

	def get_right_observation(self, view_r):
		visible = [1.0, self.ball.x, self.ball.y] if self.ball.y >= (self.window_y - view_r) else [0.0, 0.0, 0.0]
		return np.asarray(visible + [self.paddle_right.x])

	def move(self, l_move, r_move):
		# move can be 0, 1 or 2
		# 0 -> no_op
		# 1 -> down
		# 2 -> up
		if(l_move == 2):
			l_move = -1
		if(r_move == 2):
			r_move = -1
		self.paddle_left.move(l_move)
		self.paddle_right.move(r_move)
		# check for paddle and move
		# reset if something bad
		if((self.paddle_right.x <= self.ball.x and self.paddle_right.x + self.paddle_right.length - 1 >= self.ball.x and self.ball.y + self.ball.speed * self.ball.vy >= self.right_paddle_position)):
			#successfully blocked
			theta = (math.pi * 3 / 12) * ((self.ball.x - self.paddle_right.x) * 2.0 / self.paddle_right.length - 1.0)
			self.ball.vx, self.ball.vy = (math.sin(theta), -math.cos(theta))
			self.ball.move()
			return 0.0, 0.0, False

		elif((self.paddle_left.x <= self.ball.x and self.paddle_left.x + self.paddle_left.length - 1 >= self.ball.x and self.ball.y + self.ball.speed * self.ball.vy <= self.left_paddle_position)):
			theta = (math.pi * 3 / 12) * ((self.ball.x - self.paddle_left.x) * 2.0 / self.paddle_left.length - 1.0)
			self.ball.vx, self.ball.vy = (math.sin(theta), math.cos(theta))
			self.ball.move()
			return 0.0, 0.0, False


		elif((self.paddle_left.x > self.ball.x or self.paddle_left.x + self.paddle_left.length - 1 < self.ball.x) and self.ball.y + self.ball.speed * self.ball.vy <= self.left_paddle_position):
			#missed
			self.ball.move()
			self.reset()
			return -1.0, -1.0, True
		elif((self.paddle_right.x > self.ball.x or self.paddle_right.x + self.paddle_right.length - 1 < self.ball.x) and self.ball.y + self.ball.speed * self.ball.vy >= self.right_paddle_position):
			self.ball.move()
			self.reset()
			return -1.0, -1.0, True

		else:
			self.ball.move()
			return 0.0, 0.0, False

	def step(self, l_move, r_move):
		import random
		done = False
		reward_l, reward_r = 0.0, 0.0
		for _ in range(random.randint(2, 4)):
			reward_l, reward_r, done = self.move(l_move, r_move)
			if done:
				break
		return reward_l, reward_r, done

	def render(self):
		cv.imshow(self.window_name, self.get_screen())
		cv.waitKey(1)



def main():
	pong = Pong()
	for i in range(1000):
		pong.render()
		import time
		time.sleep(0.05)
		ha = 1
		if(i > 55):
			haa = 1
		else:
			haa = 0
		if(i % 10 == 0):
			ha = 0
		if(i % 2 == 0):
			haa = 0
		print(pong.step(ha, haa))

if __name__ == '__main__':
	main()
