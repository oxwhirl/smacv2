

# SMACv2 - StarCraft Multi-Agent Challenge

[SMACv2](https://github.com/oxwhirl/smacv2) is an update to [WhiRL](http://whirl.cs.ox.ac.uk)'s [SMAC](htps://github.com/oxwhirl/smac) environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on [Blizzard](http://blizzard.com)'s [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) RTS game. SMACv2 makes use of Blizzard's [StarCraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and [DeepMind](https://deepmind.com)'s [PySC2](https://github.com/deepmind/pysc2) to provide a convenient interface for autonomous agents to interact with StarCraft II, getting observations and performing actions. Unlike the [PySC2](https://github.com/deepmind/pysc2), SMACv2 concentrates on *decentralised micromanagement* scenarios, where each unit of the game is controlled by an individual RL agent.


# Quick Start

## Installing SMAC

You can install SMAC by using the following command:

```shell
pip install git+https://github.com/oxwhirl/smacv2.git
```

Alternatively, you can clone the SMAC repository and then install `smac` with its dependencies:

```shell
git clone https://github.com/oxwhirl/smacv2.git
pip install -e smac/
```

*NOTE*: If you want to extend SMAC, please install the package as follows:

```shell
git clone https://github.com/oxwhirl/smacv2.git
cd smac
pip install -e ".[dev]"
pre-commit install
```

You may also need to upgrade pip: `pip install --upgrade pip` for the install to work.

## Installing StarCraft II

SMAC is based on the full game of StarCraft II (versions >= 3.16.1). To install the game, follow the commands bellow.

### Linux

Please use the Blizzard's [repository](https://github.com/Blizzard/s2client-proto#downloads) to download the Linux version of StarCraft II. By default, the game is expected to be in `~/StarCraftII/` directory. This can be changed by setting the environment variable `SC2PATH`.

### MacOS/Windows

Please install StarCraft II from [Battle.net](https://battle.net). The free [Starter Edition](http://battle.net/sc2/en/legacy-of-the-void/) also works. PySC2 will find the latest binary should you use the default install location. Otherwise, similar to the Linux version, you would need to set the `SC2PATH` environment variable with the correct location of the game.

## SMAC maps

SMAC is composed of combat scenarios with pre-configured maps. Before SMAC can be used, these maps need to be downloaded into the `Maps` directory of StarCraft II.

You can find the maps in the `smac/env/starcraft2/maps/SMAC_Maps` directory. Copy this to your `$SC2PATH/Maps` directory.

### List the maps

To see the list of SMAC maps, together with the number of ally and enemy units and episode limit, run:

```shell
python -m smac.bin.map_list 
```

### Creating new maps

Users can extend SMAC by adding new maps/scenarios. To this end, one needs to:

- Design a new map/scenario using StarCraft II Editor:
  - Please take a close look at the existing maps to understand the basics that we use (e.g. Triggers, Units, etc),
  - We make use of special RL units which never automatically start attacking the enemy. [Here](https://docs.google.com/document/d/1BfAM_AtZWBRhUiOBcMkb_uK4DAZW3CpvO79-vnEOKxA/edit?usp=sharing) is the step-by-step guide on how to create new RL units based on existing SC2 units,
- Add the map information in [smac_maps.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/maps/smac_maps.py),
- The newly designed RL units have new ids which need to be handled in [starcraft2.py](https://github.com/oxwhirl/smac/blob/master/smac/env/starcraft2/starcraft2.py). 

## Testing SMAC

Please run the following command to make sure that `smac` and its maps are properly installed. 

```bash
python -m smac.examples.random_agents
```

## Saving and Watching StarCraft II Replays

### Saving a replay

If you’ve using our [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent RL, here’s what needs to be done:
1. **Saving models**: We run experiments on *Linux* servers with `save_model = True` (also `save_model_interval` is relevant) setting so that we have training checkpoints (parameters of neural networks) saved (click [here](https://github.com/oxwhirl/pymarl#saving-and-loading-learnt-models) for more details).
2. **Loading models**: Learnt models can be loaded using the `checkpoint_path` parameter. If you run PyMARL on *MacOS* (or *Windows*) while also setting `save_replay=True`, this will save a .SC2Replay file for `test_nepisode` episodes on the test mode (no exploration) in the Replay directory of StarCraft II. (click [here](https://github.com/oxwhirl/pymarl#watching-starcraft-ii-replays) for more details).

If you want to save replays without using PyMARL, simply call the `save_replay()` function of SMAC's StarCraft2Env in your training/testing code. This will save a replay of all epsidoes since the launch of the StarCraft II client.

The easiest way to save and later watch a replay on Linux is to use [Wine](https://www.winehq.org/).

### Watching a replay

You can watch the saved replay directly within the StarCraft II client on MacOS/Windows by *clicking on the corresponding Replay file*.

You can also watch saved replays by running:

```shell
python -m pysc2.bin.play --norender --replay <path-to-replay>
```

This works for any replay as long as the map can be found by the game. 

For more information, please refer to [PySC2](https://github.com/deepmind/pysc2) documentation.

# Documentation 

For the detailed description of the environment, read the [SMACv2 documentation](docs/smac.md). 

# Code Examples

Below is a small code example which illustrates how SMACv2 can be used. Here, individual agents execute random policies after receiving the observations and global state from the environment.  

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
