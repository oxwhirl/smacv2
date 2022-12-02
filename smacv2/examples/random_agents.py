from __future__ import absolute_import, division, print_function

import time
from os import replace

import numpy as np
from absl import logging
from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)


def main():

    distribution_config = {
        "n_units": 10,
        "n_enemies": 11,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
            "exception_unit_types": ["medivac"],
        },
        
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        }
        
    }
    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=False,
        conic_fov=False,
        use_unit_ranges=True,
        min_attack_range=2,
        obs_own_pos=True,
        fully_observable=False,
    )

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    cap_size = env_info["cap_shape"]

    n_episodes = 10
    print("Training episodes")
    env.reset()
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0
        state_features = env.get_state_feature_names()
        obs_features = env.get_obs_feature_names()

        while not terminated:
            obs = env.get_obs()
            print(f"Obs size: {obs[0].shape}")
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
        assert len(state) == len(state_features)
        assert len(obs[0]) == len(obs_features)


if __name__ == "__main__":
    main()
