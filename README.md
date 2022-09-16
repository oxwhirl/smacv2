# SMACv2 Documentation

# Introduction

SMACv2 is an update to [Whirl’s](https://whirl.cs.ox.ac.uk/) [Starcraft Multi-Agent Challenge](https://github.com/oxwhirl/smac), which is a benchmark for research in the field of cooperative multi-agent reinforcement learning. SMAC and SMACv2 both focus on decentralised micromanagement scenarios in [StarCraft II](https://starcraft2.com/en-gb/), rather than the full game. It makes use of Blizzard’s StarCraft II Machine Learning API as well as Deepmind’s PySC2. We hope that you will enjoy using SMACv2! More details about SMAC can be found in the [SMAC README](https://github.com/oxwhirl/smac/blob/master/README.md) as well as the [SMAC paper](https://arxiv.org/abs/1902.04043). **SMAC retains exactly the same API as SMAC so you should not need to change your algorithm code other than adjusting to the new observation and state size**.

If you encounter difficulties using SMACv2, or have suggestions please raise an issue, or better yet, open a pull request! If you make changes to the SMACv2 state or observation space as part of your work, and believe they help performance, please do so as a wrapper around SMACv2 and open a pull request to allow others to benefit from your work 🙂. 

The aim of this README is to answer some basic technical questions and to get people started with SMACv2. For a more scientific account of the work of developing the benchmark, please read our paper!

# Differences To SMAC

SMACv2 makes three major changes to SMACv2: randomising start positions, randomising unit types, and restricting the agent field-of-view and shooting range to a cone. These first two changes were motivated by the discovery that many maps in SMAC lack enough randomness to challenge contemporary MARL algorithms. The final change requires agents to explore their environment and effectively gather information. For more details on the motivation behind these changes, please check the accompanying paper, where these are discussed in much more detail!

## Capability Config

All the procedurally generated content in SMACv2 is managed through the **Capability Config.** This describes what units are generated and in what positions. The presence of keys in this config tells SMACv2 that a certain environment component is generated or not. As an example, consider the below config:

```yaml
capability_config:
    n_units: 5
    team_gen:
      dist_type: "weighted_teams"
      unit_types: 
        - "marine"
        - "marauder"
        - "medivac"
      weights:
        - 0.45
        - 0.45
        - 0.1
      exception_unit_types:
        - "medivac"
      observe: True

    start_positions:
      dist_type: "surrounded_and_reflect"
      p: 0.5
      n_enemies: 5
      map_x: 32
      map_y: 32
```

This config is the default config for the SMACv2 Terran scenarios. The `start_positions` key tells SMACv2 to randomly generate start positions. Similarly the `team_gen` key tells SMACv2 to randomly generate teams. The `dist_type` tells SMACv2 **how** to generate some content. For example, team generation has the key `weighted_teams` , where each unit type is spawned with a certain weight. In this case a Stalker is spawned with probability `0.45` for example. Don’t worry too much about the other options for now — they are distribution-specific.

All the distributions are implemented in the [distributions.py](https://github.com/oxwhirl/smacv2/blob/main/smac/env/starcraft2/distributions.py) file. We encourage users to contribute their own keys and distributions for procedurally generated content!

## Random Start Positions

Random start positions come in two different types. First, there is the `surrounded` type, where the allied units are spawned in the middle of the map, and surrounded by enemy units. An example is shown below.

![surrounded.png](docs/imgs/surrounded.png)

This challenges the allied units to overcome the enemies approach from multiple angles at once. Secondly, there are the `reflect_position` scenarios. These randomly select positions for the allied units, and then reflect their positions in the midpoint of the map to get the enemy spawn positions. For example see the image below.

![surrounded.png](docs/imgs/reflect.png)

The probability of one type of scenario or the other is controlled with the `p` setting in the capability config. The cones are not visible in the above screenshot because they have not spawned in yet. 

## Random Unit Types

Battles in SMACv2 do not always feature units of the same type each time, as they did in SMAC. Instead, units are spawned randomly according to certain pre-fixed probabilities. Units in StarCraft II are split up into different *races.* Units from different races cannot be on the same team. For each of the three races (Protoss, Terran, and Zerg), SMACv2 uses three unit types.

| Race | Unit | Generation Probability |
| --- | --- | --- |
| Terran | Marine | 0.45 |
|  | Marauder | 0.45 |
|  | Medivac | 0.1 |
| Protoss | Stalker | 0.45 |
|  | Zealot | 0.45 |
|  | Colossus | 0.1 |
| Zerg | Zergling | 0.45 |
|  | Hydralisk | 0.45 |
|  | Baneling | 0.1 |

Each race has a unit that is generated less often than the others. These are for different reasons. Medivacs are healing-only units and so an abundance of them leads to strange, very long scenarios. Colossi are very powerful units and over-generating them leads to battles being solely determined by colossus use. Banelings are units that explode. If they are too prevalent, the algorithm learns to hide in the corner and hope the enemies all explode!

These weights are all controllable via the `capability_config` . However, if you do decide to change them we recommend that you do some tests to check that the scenarios you have made are sensible! Weights changes can sometimes have unexpected consequences.

## Field-Of-View

The field of view in SMACv2 is restricted to a cone. The units must move this cone around to allow them to view and target their enemies. As in SMAC, the unit sight and shoot ranges are fixed. There are extra actions which allow agents to ‘snap’ their cone to a specific point around a circle. These actions are always available. 

The cones are displayed in the example screenshots above using dotted lines. The conic field-of-view is controlled with the `conic_fov` option.

# Getting Started

This section will take you through the basic set-up of SMACv2. The set-up process has changed very little from the process for SMAC, so if you are familiar with that, follow the steps as you usually would. Make sure you have the `32x32_flat.SC2Map` map file in your `SMAC_Maps` folder. 

First, you will need to install StarCraft II. On windows or mac, follow the instructions on the [StarCraft website](https://starcraft2.com/en-gb/). For linux, you can use the bash script [here](https://github.com/benellis3/mappo/blob/main/install_sc2.sh). Then copy 

Then simply install SMAC as a package:

```bash
pip install git+https://github.com/oxwhirl/smacv2.git
```

[NOTE]: If you want to extend SMACv2, you must install it like this:

```bash
git clone https://github.com/oxwhirl/smacv2.git
cd smac
pip install -e ".[dev]"
pre-commit install
```

If you tried these instructions and couldn’t get SMACv2 to work, please let us know by raising an issue. 

# Modifying SMACv2

SMACv2 procedurally generates some content. We encourage everyone to modify and expand upon the procedurally generated content in SMACv2. 

Procedurally generated content conceptually has two parts: a distribution and an implementation. The implementation part lives in the [starcraft2.py](https://github.com/oxwhirl/smacv2/blob/main/smac/env/starcraft2/starcraft2.py) file and should handle actually generating whatever content is required (e.g. the spawning units at the correct start positions) using the StarCraft APIs given a config passed in at the start of the episode to the `reset` function. 

The second part is the distribution. These live in [distributions.py](https://github.com/oxwhirl/smacv2/blob/main/smac/env/starcraft2/distributions.py) and specify the distribution the content is generated according to. For example start positions might be generated randomly across the whole map. The `distributions.py` file contains a few examples of distributions for the already implemented generated content in SMAC.

# Code Example

SMACv2 follows the same API as SMAC and so can be used exactly the same way. As an example, the below code allows individual agents to execute random policies. The config corresponds to the 5 unit Terran map from SMACv2. 

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os import replace

from smac.env import StarCraft2Env
import numpy as np
from absl import logging
import time

from smac.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

logging.set_verbosity(logging.DEBUG)

def main():

    distribution_config = {
        "n_units": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }
    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=True,
        conic_fov=True,
        obs_own_pos=True,
    )

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]

    n_episodes = 10

    print("Training episodes")
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
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
        print("Total reward in episode {} = {}".format(e, episode_reward))

if __name__ == "__main__":
    main()
```

# FAQ

### Why are the FOV cones not visible in my replay?

Unfortunately, we draw the FOV cones using the `DebugDraw` command in the StarCraft II API. This is not recorded in replays. There is nothing we can do to fix this to our knowledge 😟.  If you instead collect episodes for replays on a machine with a viewable StarCraft game, you will be able to see the FOV cones. If you know how we can fix this, please get in touch by raising an issue, we’d love to hear from you!

### Why do SMAC maps not work in SMACv2?

For now, SMAC is not backwards compatible with old SMAC maps, although we will implement this if there is enough demand.

# Questions/Comments

If you have any questions or suggestions either raise an issue in this repo or email [Ben Ellis](mailto:benellis@robots.ox.ac.uk) and we will try our
best to answer your query.
