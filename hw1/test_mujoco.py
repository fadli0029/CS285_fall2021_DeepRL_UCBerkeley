# use this to test your mujoco environment
# if you run into error like this:
# GLEW initalization error: Missing GL version,
# then check this link out: https://github.com/openai/mujoco-py/issues/268
import gym
env = gym.make('Ant-v2')
observation = env.reset()
for _ in range(1000):
   env.render()
   action = env.action_space.sample()
   observation, reward, done, info = env.step(action)
   if done:
      observation= env.reset()
env.close()
