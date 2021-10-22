from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool
from numpy.random import choice
from lib import Logger, VideoRecorder, save_json
import matplotlib.pyplot as plt


def act(env, observations):
    actions = dict()

    for agent_handle, observation in enumerate(observations):
        print(env.action_space)
        action = choice([
            RailEnvActions.MOVE_FORWARD,
            # RailEnvActions.MOVE_RIGHT,
            # RailEnvActions.MOVE_LEFT,
            # RailEnvActions.STOP_MOVING
        ])
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
logger = Logger(env)

fig = plt.figure()
rows = 14
columns = 14

for step in range(rows * columns):
    actions = act(env, observations)
    next_observations, rewards, dones, info = env.step(actions)

    logger.log(next_observations, rewards)

    rail_map = [[sum(cell) for cell in row] for row in next_observations[0][0]]

    fig.add_subplot(rows, columns, step + 1)

    plt.imshow(rail_map)

    if dones['__all__'] == True:
        break

    for agent_handle in env.get_agent_handles():
        score += rewards[agent_handle]

    print('Timestep {}, total score = {}'.format(step, score))

plt.show()
logger.save()
