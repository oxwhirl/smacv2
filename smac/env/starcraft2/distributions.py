from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict
from itertools import combinations_with_replacement
from random import choice, shuffle
from math import inf
from numpy.random import default_rng


class Distribution(ABC):
    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        pass

    @property
    @abstractproperty
    def n_tasks(self) -> int:
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

    def generate(self):
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

        self.combinations = list(
            combinations_with_replacement(self.units, self.n_units)
        )

    def generate(self):
        team = list(choice(self.combinations))
        team_id = self.combinations.index(tuple(team))
        shuffle(team)
        return {"team_gen": {"item": team, "id": team_id}}

    @property
    def n_tasks(self):
        return len(self.combinations)


register_distribution("all_teams", AllTeamsDistribution)


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

    def generate(self):
        probs = self.rng.uniform(
            low=self.lower_bound, high=self.upper_bound, size=(self.n_units,)
        )
        return {self.env_key: {"item": probs, "id": 0}}

    @property
    def n_tasks(self):
        return inf


register_distribution("per_agent_uniform", PerAgentUniformDistribution)


class MaskDistribution(Distribution):
    def __init__(self, config):
        self.config = config
        self.mask_probability = config["mask_probability"]
        self.n_units = config["n_units"]
        self.n_enemies = config["n_enemies"]
        self.rng = default_rng()

    def generate(self):
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
