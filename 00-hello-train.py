from PIL import Image
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.utils.rendertools import RenderTool
from datetime import datetime

env = RailEnv(
    width=24,
    height=24,
    number_of_agents=1,
    rail_generator=sparse_rail_generator(max_num_cities=2),
    line_generator=sparse_line_generator(),
    obs_builder_object=GlobalObsForRailEnv()
)
observations, info = env.reset()

render_tool = RenderTool(env, gl="PILSVG")
render_tool.render_env()
path = str(datetime.now().timestamp()) + '.png'
Image.fromarray(render_tool.get_image()).save(path)