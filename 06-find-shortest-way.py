from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_env import RailEnv
import numpy as np
import matplotlib.pyplot as plt

env = RailEnv(
    width=40, 
    height=40,
)
obs = env.reset()

obs, rew, done, info = env.step({
    0: np.random.randint(0, 5),
    1: np.random.randint(0, 5)
})

for agent_handle in env.get_agent_handles():
    agent_obs = obs[agent_handle]
    
    #rail_map = [[np.sum(cell) for cell in row] for row in agent_obs[0]]
    #plt.matshow(rail_map)
    #plt.show()
    
    agent_states = np.transpose(agent_obs[1], (2, 0, 1))
    agent_targets = np.transpose(agent_obs[2], (2, 0, 1))
    
    print('- Agent position\n')
    plt.matshow(agent_states[0])
    
    print('- Current agent:')
    plt.matshow(agent_targets[0])
    plt.show()