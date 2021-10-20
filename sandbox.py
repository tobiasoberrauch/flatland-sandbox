from PIL import Image
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool
from numpy.random import randint
from datetime import datetime
import cv2

writer = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800, 600))
writer.write()
writer.release()

def screenshot(env):
    render_tool = RenderTool(env, gl="PILSVG")
    render_tool.render_env()
    path = str(datetime.now().timestamp()) + '.png'
    Image.fromarray(render_tool.get_image()).save(path)

def act(env, observations):
    actions = dict()

    for agent_handle, observation in enumerate(observations):
        action = randint(env.action_space[0])
        actions.update({agent_handle: action})
    
    return actions

env = RailEnv(
    width=24,
    height=24,
    number_of_agents=1,
    rail_generator=sparse_rail_generator(max_num_cities=2),
    line_generator=sparse_line_generator(),
    obs_builder_object=GlobalObsForRailEnv()
)

observations, info = env.reset()
score = 0
actions = dict()

for step in range(50):
    actions = act(env, observations)
    print('action', actions)
    next_observations, rewards, dones, info = env.step(actions)
    print('rewards', rewards)
    
    for agent_handle in env.get_agent_handles():
        score += rewards[agent_handle]

    screenshot(env)
    print('Timestep {}, total score = {}'.format(step, score))