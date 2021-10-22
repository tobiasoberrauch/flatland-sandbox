from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.observations import Node
from flatland.envs.rail_env import RailEnv
# for graphs:
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def shortest_way(G, node, path):
    d = {neibor:G.nodes[neibor]['object']['dist_min_to_target'] for neibor in G.neighbors(node)}
    print(node, end='')
    print(' -> ', end='')
    print(d)
    # exhausted the depth?
    if len(d) == 0:
        return path
    # choose the shortes path neighbor and continue recursively
    best_neibor = min(d, key=d.get)
    return shortest_way(G, best_neibor, path + [best_neibor])

def node_data(node, direction):
    return {
        "direction" : direction,
        "dist_own_target_encountered" : node.dist_own_target_encountered,
        "dist_other_target_encountered" : node.dist_other_target_encountered,
        "dist_other_agent_encountered" : node.dist_other_agent_encountered,
        "dist_potential_conflict" : node.dist_potential_conflict,
        "dist_unusable_switch" : node.dist_unusable_switch,
        "dist_to_next_branch" : node.dist_to_next_branch,
        "dist_min_to_target" : node.dist_min_to_target,
        "num_agents_same_direction" : node.num_agents_same_direction,
        "num_agents_opposite_direction" : node.num_agents_opposite_direction,
        "num_agents_malfunctioning" : node.num_agents_malfunctioning,
        "speed_min_fractional" : node.speed_min_fractional,
        "num_agents_ready_to_depart" : node.num_agents_ready_to_depart
    }

env = RailEnv(width=30,
              height=30,
              number_of_agents=3,
              rail_generator=sparse_rail_generator(),
              obs_builder_object=TreeObsForRailEnv(max_depth=5)
              )

env.reset()


obs, rew, done, info = env.step({
    0: np.random.randint(0, 5),
    1: np.random.randint(0, 5),
    2: np.random.randint(0, 5)
})

# screenshot

G = nx.DiGraph()
node = node_data(obs[0], None)
print(node)
G.add_node(1, object=node)

last_id = 1
fringe = [(obs[0], last_id)]

while len(fringe) > 0:
  parent, parent_id = fringe.pop()
  for key, node in parent.childs.items():
    if type(node) is Node:
      last_id += 1
      G.add_node(last_id, object=node_data(node, key))
      G.add_edge(parent_id, last_id)

      fringe.append((node, last_id))


plt.figure()
nx.draw_planar(G, with_labels=True)
plt.show()

print(shortest_way(G, 1, []))