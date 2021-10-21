from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool
from numpy.random import randint
from lib import VideoRecorder


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
video_recorder = VideoRecorder()
render_tool = RenderTool(env, gl="PILSVG")

for step in range(200):
    actions = act(env, observations)
    next_observations, rewards, dones, info = env.step(actions)
    print('next_observations', next_observations)

    if dones['__all__'] == True:
        break

    for agent_handle in env.get_agent_handles():
        score += rewards[agent_handle]

    render_tool.render_env()
    video_recorder.add_image(render_tool.get_image())
    print('Timestep {}, total score = {}'.format(step, score))

video_recorder.save()
