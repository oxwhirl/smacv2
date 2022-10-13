from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from typing import Any, Dict
from itertools import combinations_with_replacement
from random import choice, shuffle
from math import inf
from numpy.random import default_rng
import numpy as np


class Distribution(ABC):
    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        pass

    @property
    @abstractproperty
    def n_tasks(self) -> int:
        pass


class EvolutionaryDistribution(ABC):
    @abstractmethod
    def perturb(self, episode_config: Dict[str, Any]) -> Dict[str, Any]:
        pass


DISTRIBUTION_MAP = {}


def get_distribution(key):
    return DISTRIBUTION_MAP[key]


def register_distribution(key, cls):
    DISTRIBUTION_MAP[key] = cls


class FixedDistribution(Distribution):
    """A generic disribution that draws from a fixed list.
    May operate in test mode, where items are drawn sequentially,
    or train mode where items are drawn randomly. Example uses of this
    are for team generation or per-agent accuracy generation in SMAC by
    drawing from separate fixed lists at test and train time.
    """

    def __init__(self, config):
        """
        Args:
            config (dict): Must contain `env_key`, `test_mode` and `items`
            entries. `env_key` is the key to pass to the environment so that it
            recognises what to do with the list. `test_mode` controls the sampling
            behaviour (sequential if true, uniform at random if false), `items`
            is the list of items (team configurations/accuracies etc.) to sample from.
        """
        self.config = config
        self.env_key = config["env_key"]
        self.test_mode = config["test_mode"]
        self.teams = config["items"]
        self.index = 0

    def generate(self) -> Dict[str, Dict[str, Any]]:
        """Returns:
        Dict: Returns a dict of the form
        {self.env_key: {"item": <item>, "id": <item_index>}}
        """
        if self.test_mode:
            team = self.teams[self.index]
            team_id = self.index
            self.index = (self.index + 1) % len(self.teams)
            shuffle(team)
            return {self.env_key: {"item": team, "id": team_id}}
        else:
            team = choice(self.teams)
            team_id = self.teams.index(team)
            shuffle(team)
            return {self.env_key: {"item": team, "id": team_id}}

    @property
    def n_tasks(self):
        return len(self.teams)


register_distribution("fixed", FixedDistribution)


class AllTeamsDistribution(Distribution):
    def __init__(self, config):
        self.config = config
        self.units = config["unit_types"]
        self.n_units = config["n_units"]
        self.exceptions = config.get("exception_unit_types", [])
        self.env_key = config["env_key"]
        self.combinations = list(
            combinations_with_replacement(self.units, self.n_units)
        )

    def generate(self) -> Dict[str, Dict[str, Any]]:
        team = []
        while not team or all(member in self.exceptions for member in team):
            team = list(choice(self.combinations))
            team_id = self.combinations.index(tuple(team))
            shuffle(team)
        return {
            self.env_key: {
                "ally_team": team,
                "enemy_team": team,
                "id": team_id,
            }
        }

    @property
    def n_tasks(self):
        # TODO adjust so that this can handle exceptions
        assert not self.exceptions
        return len(self.combinations)


register_distribution("all_teams", AllTeamsDistribution)


class WeightedTeamsDistribution(Distribution, EvolutionaryDistribution):
    def __init__(self, config):
        self.config = config
        self.units = np.array(config["unit_types"])
        self.n_units = config["n_units"]
        self.n_enemies = config["n_enemies"]
        assert (
            self.n_enemies >= self.n_units
        ), "Only handle larger number of enemies than allies"
        self.weights = np.array(config["weights"])
        # unit types that cannot make up the whole team
        self.exceptions = config.get("exception_unit_types", set())
        self.rng = default_rng()
        self.env_key = config["env_key"]

    def _gen_team(self, n_units: int, use_exceptions: bool):
        team = []
        while not team or (
            all(member in self.exceptions for member in team)
            and use_exceptions
        ):
            team = list(
                self.rng.choice(self.units, size=(n_units,), p=self.weights)
            )
            shuffle(team)
        return team

    def generate(self) -> Dict[str, Dict[str, Any]]:
        team = self._gen_team(self.n_units, use_exceptions=True)
        enemy_team = team.copy()
        if self.n_enemies > self.n_units:
            extra_enemies = self._gen_team(
                self.n_enemies - self.n_units, use_exceptions=True
            )
            enemy_team.extend(extra_enemies)

        return {
            self.env_key: {
                "ally_team": team,
                "enemy_team": enemy_team,
                "id": 0,
            }
        }

    def perturb(self, episode_config: Dict[str, Any]):
        ally_team = episode_config[self.env_key]["ally_team"]
        enemy_team = episode_config[self.env_key]["enemy_team"]
        # as in some other methods, uniformly choose a unit on the
        # field to change the unit type of
        unit = self.rng.integers(0, self.n_units + self.n_enemies)
        unit_type = (
            ally_team[unit]
            if unit < self.n_units
            else enemy_team[unit - self.n_units]
        )
        other_unit_types = self.units[self.units != unit_type]
        new_unit_type = self.rng.choice(other_unit_types)
        # change the appropriate unit types
        if unit >= self.n_units:
            unit = unit - self.n_units
        enemy_team[unit] = new_unit_type
        if unit < self.n_units:
            ally_team[unit] = new_unit_type
        return {
            self.env_key: {
                "ally_team": ally_team,
                "enemy_team": enemy_team,
                "id": 0,
            }
        }

    @property
    def n_tasks(self):
        return inf


register_distribution("weighted_teams", WeightedTeamsDistribution)


class PerAgentUniformDistribution(Distribution):
    """A generic distribution for generating some information per-agent drawn
    from a uniform distribution in a specified range.
    """

    def __init__(self, config):
        self.config = config
        self.lower_bound = config["lower_bound"]
        self.upper_bound = config["upper_bound"]
        self.env_key = config["env_key"]
        self.n_units = config["n_units"]
        self.rng = default_rng()

    def generate(self) -> Dict[str, Dict[str, Any]]:
        probs = self.rng.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(self.n_units, len(self.lower_bound)),
        )
        return {self.env_key: {"item": probs, "id": 0}}

    @property
    def n_tasks(self):
        return inf


register_distribution("per_agent_uniform", PerAgentUniformDistribution)


class MaskDistribution(Distribution):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mask_probability = config["mask_probability"]
        self.n_units = config["n_units"]
        self.n_enemies = config["n_enemies"]
        self.rng = default_rng()

    def generate(self) -> Dict[str, Dict[str, Any]]:
        mask = self.rng.choice(
            [0, 1],
            size=(self.n_units, self.n_enemies),
            p=[
                self.mask_probability,
                1.0 - self.mask_probability,
            ],
        )
        return {"enemy_mask": {"item": mask, "id": 0}}

    @property
    def n_tasks(self):
        return inf


register_distribution("mask", MaskDistribution)


class ReflectPositionDistribution(Distribution, EvolutionaryDistribution):
    """Distribution that will generate enemy and ally
    positions. Generates ally positions uniformly at
    random and then reflects these in a vertical line
    half-way across the map to get the enemy positions.
    Only works when the number of agents and enemies is the same.
    """

    def __init__(self, config):
        self.config = config
        self.n_units = config["n_units"]
        self.n_enemies = config["n_enemies"]
        assert (
            self.n_enemies >= self.n_units
        ), "Number of enemies must be >= number of units"
        self.map_x = config["map_x"]
        self.map_y = config["map_y"]
        config_copy = deepcopy(config)
        config_copy["env_key"] = "ally_start_positions"
        config_copy["lower_bound"] = (0, 0)
        # subtract one from the x coordinate because SC2 goes wrong
        # when you spawn ally and enemy units on top of one another
        # -1 gives a sensible 'buffer zone' of size 2
        config_copy["upper_bound"] = (self.map_x / 2 - 1, self.map_y)
        self.pos_generator = PerAgentUniformDistribution(config_copy)
        self.rng = default_rng()
        if self.n_enemies > self.n_units:
            enemy_config_copy = deepcopy(config)
            enemy_config_copy["env_key"] = "enemy_start_positions"
            enemy_config_copy["lower_bound"] = (self.map_x / 2, 0)
            enemy_config_copy["upper_bound"] = (self.map_x, self.map_y)
            enemy_config_copy["n_units"] = self.n_enemies - self.n_units
            self.enemy_pos_generator = PerAgentUniformDistribution(
                enemy_config_copy
            )

    def generate(self) -> Dict[str, Dict[str, Any]]:
        ally_positions_dict = self.pos_generator.generate()
        ally_positions = ally_positions_dict["ally_start_positions"]["item"]
        enemy_positions = np.zeros((self.n_enemies, 2))
        enemy_positions[: self.n_units, 0] = self.map_x - ally_positions[:, 0]
        enemy_positions[: self.n_units, 1] = ally_positions[:, 1]
        if self.n_enemies > self.n_units:
            gen_enemy_positions = self.enemy_pos_generator.generate()
            gen_enemy_positions = gen_enemy_positions["enemy_start_positions"][
                "item"
            ]
            enemy_positions[self.n_units :, :] = gen_enemy_positions
        return {
            "ally_start_positions": {"item": ally_positions, "id": 0},
            "enemy_start_positions": {"item": enemy_positions, "id": 0},
        }

    def perturb(
        self, episode_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        ally_positions = episode_config["ally_start_positions"]["item"]
        enemy_positions = episode_config["enemy_start_positions"]["item"]
        # Because there can be more enemies than allies, we need to
        # choose a unit *among all the units*, and if it has an
        # equivalent ally, move that. This is fairer than just choosing
        # an enemy, where there is a greater probability to pick one of the
        # enemies not matched with an ally.
        unit = self.rng.integers(0, self.n_units + self.n_enemies)
        new_positions = self.pos_generator.generate()["ally_start_positions"][
            "item"
        ]
        if unit > self.n_units:
            unit = unit - self.n_units
        if unit < self.n_units:
            # generate new ally position
            ally_positions[unit] = new_positions[0]
            enemy_positions[unit, 0] = self.map_x - ally_positions[unit, 0]
            enemy_positions[unit, 1] = ally_positions[unit, 1]
        else:
            new_positions = self.pos_generator.generate()[
                "ally_start_positions"
            ]["item"]
            enemy_positions[unit, 0] = self.map_x - new_positions[0, 0]
            enemy_positions[unit, 1] = new_positions[0, 1]
        episode_config["ally_start_positions"] = {
            "item": ally_positions,
            "id": 0,
        }
        episode_config["enemy_start_positions"] = {
            "item": enemy_positions,
            "id": 0,
        }
        return episode_config

    @property
    def n_tasks(self) -> int:
        return inf


register_distribution("reflect_position", ReflectPositionDistribution)


class SurroundedPositionDistribution(Distribution, EvolutionaryDistribution):
    """Distribution that generates ally positions in a
    circle at the centre of the map, and then has enemies
    randomly distributed in the four diagonal directions at a
    random distance.
    """

    def __init__(self, config):
        self.config = config
        self.n_units = config["n_units"]
        self.n_enemies = config["n_enemies"]
        self.map_x = config["map_x"]
        self.map_y = config["map_y"]
        self.perturb_threshold = config.get("perturb_threshold", 0.5)
        self.rng = default_rng()
        self.offset = 2
        self.diagonal_to_centre_point = {
            0: np.array(
                [self.map_x / 2 - self.offset, self.map_y / 2 - self.offset]
            ),
            1: np.array(
                [self.map_x / 2 - self.offset, self.map_y / 2 + self.offset]
            ),
            2: np.array(
                [self.map_x / 2 + self.offset, self.map_y / 2 - self.offset]
            ),
            3: np.array(
                [self.map_x / 2 + self.offset, self.map_y / 2 + self.offset]
            ),
        }
        self.diagonal_to_point_map = {
            0: np.array([0, 0]),
            1: np.array([0, self.map_y]),
            2: np.array([self.map_x, 0]),
            3: np.array([self.map_x, self.map_y]),
        }

    def generate(self) -> Dict[str, Dict[str, Any]]:
        # need multiple centre points because SC2 does not cope with
        # spawning ally and enemy units on top of one another in some
        # cases
        centre_point = np.array([self.map_x / 2, self.map_y / 2])

        ally_position = np.tile(centre_point, (self.n_units, 1))
        enemy_position = np.zeros((self.n_enemies, 2))
        # decide on the number of groups (between 1 and 4)
        n_groups = self.rng.integers(1, 5)
        # generate the number of enemies in each group
        group_membership = self.rng.multinomial(
            self.n_enemies, np.ones(n_groups) / n_groups
        )
        # decide on the distance along the diagonal for each group
        group_position = self.rng.uniform(size=(n_groups,))
        group_diagonals = self.rng.choice(
            np.array(range(4)), size=(n_groups,), replace=False
        )

        unit_index = 0
        for i in range(n_groups):
            t = group_position[i]
            enemy_position[
                unit_index : unit_index + group_membership[i], :
            ] = self.diagonal_to_centre_point[
                group_diagonals[i]
            ] * t + self.diagonal_to_point_map[
                group_diagonals[i]
            ] * (
                1 - t
            )
            unit_index += group_membership[i]

        return {
            "ally_start_positions": {"item": ally_position, "id": 0},
            "enemy_start_positions": {"item": enemy_position, "id": 0},
        }

    def perturb(self, episode_config: Dict[str, Any]) -> Dict[str, Any]:
        enemy_positions = episode_config["enemy_start_positions"]["item"]
        enemy_groups = np.unique(enemy_positions, axis=0)
        # decide whether to move a unit or a group
        p_move_unit = self.rng.uniform()
        if p_move_unit < self.perturb_threshold:
            # move a unit to a different group
            enemy_index = self.rng.integers(0, self.n_enemies)
            enemy_pos = enemy_positions[enemy_index]
            # find out what group the randomly chosen enemy is in,
            # and check that we only get 1 correct match in the groups
            enemy_group_indices = np.where(enemy_groups == enemy_pos)
            assert len(enemy_group_indices) == 2
            assert enemy_group_indices[0][0] == enemy_group_indices[1][0]
            # choose a new group to be in
            enemy_group = enemy_group_indices[0][0]
            allowed_groups = np.delete(enemy_groups, enemy_group, axis=0)
            new_pos_index = self.rng.integers(0, allowed_groups.shape[0])
            new_pos = allowed_groups[new_pos_index]
            enemy_positions[enemy_index] = new_pos
            episode_config["enemy_start_positions"] = {
                "item": enemy_positions,
                "id": 0,
            }
        else:
            # move a group to a new location
            group_index = self.rng.integers(0, enemy_groups)
            group_pos = enemy_groups[group_index]
            # work out which quadrant we are in
            in_right_half = group_pos[0] > self.map_x / 2
            in_top_half = group_pos[1] > self.map_y / 2
            diagonal_index = 2 * in_right_half + in_top_half
            # generate a new position for the group
            t = self.rng.uniform()
            new_group_position = self.diagonal_to_centre_point[
                diagonal_index
            ] * t + self.diagonal_to_point_map[diagonal_index] * (1 - t)
            # switching up positions
            mask = (enemy_positions == group_pos).all(axis=1)
            enemy_positions[mask] = new_group_position
            episode_config["enemy_start_positions"] = {
                "item": enemy_positions,
                "id": 0,
            }
        return episode_config

    @property
    def n_tasks(self):
        return inf


register_distribution("surrounded", SurroundedPositionDistribution)

# If this becomes common, then should work on a more satisfying way
# of doing this
class SurroundedAndReflectPositionDistribution(
    Distribution, EvolutionaryDistribution
):
    def __init__(self, config):
        self.p_threshold = config["p"]
        self.surrounded_distribution = SurroundedPositionDistribution(config)
        self.reflect_distribution = ReflectPositionDistribution(config)
        self.rng = default_rng()
        self.map_x = config["map_x"]
        self.map_y = config["map_y"]

    def generate(self) -> Dict[str, Dict[str, Any]]:
        p = self.rng.random()
        if p > self.p_threshold:
            return self.surrounded_distribution.generate()
        else:
            return self.reflect_distribution.generate()

    def perturb(self, episode_config: Dict[str, Any]) -> Dict[str, Any]:
        ally_positions = episode_config["ally_start_positions"]["item"]
        if np.all(
            ally_positions == np.array([self.map_x / 2, self.map_y / 2])
        ):
            return self.surrounded_distribution.perturb(episode_config)
        else:
            return self.reflect_distribution.perturb(episode_config)

    @property
    def n_tasks(self):
        return inf


register_distribution(
    "surrounded_and_reflect", SurroundedAndReflectPositionDistribution
)
