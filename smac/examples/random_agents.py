from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

from smac.env import StarCraft2Env
import numpy as np
from absl import logging
import time

logging.set_verbosity(logging.DEBUG)


# ally_train_teams = [
#     ["marine"] * 6 + ["marauder"] * 4,
#     ["marine"] * 10,
# ]
# ally_test_teams = [
#     ["marine"] * 4 + ["marauder"] * 3 + ["medivac"] * 3
# ]

# ally_train_teams = [["hydralisk"] * 10]
# ally_test_teams = [["zergling"] * 10]
ally_train_teams = [["stalker"] * 10, ["zealot"] * 10]
ally_test_teams = [["stalker"] * 5 + ["zealot"] * 5]


def main():

    capability_config = {
        # "enemy_mask": {},
        # "team_gen": {"observe": True, "n_units": 5},
        "health": {"observe": True},
    }
    env = StarCraft2Env(
        map_name="10gen_protoss",
        capability_config=capability_config,
        debug=True,
    )
    team = ["stalker", "zealot", "stalker", "zealot", "zealot"]
    enemy_mask = np.array(
        [
            [0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=np.int64,
    )
    attack_probs = np.array(
        [0.8, 0.8, 0.6, 0.9, 0.99, 0.25, 0.9, 0.122, 0.66, 0.98]
    )
    health_levels = attack_probs
    # env.reset()

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    cap_size = env_info["cap_shape"]

    n_episodes = 5

    print("Training episodes")
    for e in range(n_episodes):
        env.reset(
            {
                # "team_gen": {"item": team, "id": 0},
                # "enemy_mask": {"item": enemy_mask, "id": 0},
                "health": {"item": health_levels, "id": 0},
            }
        )
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


if __name__ == "__main__":
    main()
