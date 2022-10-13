from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

import numpy as np
from absl import logging
import time
from smac.env.starcraft2.adversarial import AdversarialSC2Env

from smac.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)


def main():

    distribution_config = {
        "n_units": 4,
        "n_enemies": 4,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot"],
            "weights": [0.5, 0.5],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    env = AdversarialSC2Env(
        capability_config=distribution_config,
        map_name="10gen_protoss",
        debug=True,
        conic_fov=False,
        use_unit_ranges=True,
        fully_observable=False,
    )
    # env.reset()

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    cap_size = env_info["cap_shape"]

    n_episodes = 5

    env.reset_random()
    print("Training episodes")
    for e in range(n_episodes):
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            cap = env.get_capabilities()
            # env.render()  # Uncomment for rendering

            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            time.sleep(0.15)
            episode_reward += reward

        # print("Total reward in episode {} = {}".format(e, episode_reward))
        env.mutate_level(2)


if __name__ == "__main__":
    main()
