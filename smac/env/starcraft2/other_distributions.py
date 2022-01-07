from math import dist
from re import I
import numpy as np
from random import choice, shuffle

DISTRIBUTIONS = {}


def get_distribution(distribution_key):
    return DISTRIBUTIONS[distribution_key]


def register_distribution(key, fn):
    DISTRIBUTIONS[key] = fn


def stub_distribution_function(n_agents, **kwargs):
    def distribution():
        return [0.0] * n_agents

    return distribution


register_distribution("stub", stub_distribution_function)


def uniform_distribution(n_agents, **kwargs):
    rng = np.random.default_rng()

    def distribution():
        return rng.uniform(
            low=kwargs["attack_probability_low"],
            high=kwargs["attack_probability_high"],
            size=(n_agents,),
        )

    return distribution


register_distribution("uniform", uniform_distribution)


def fixed_attack_distribution(n_agents, **kwargs):
    distributions = list(kwargs["distributions"])

    def get_distribution():
        team = list(choice(distributions))
        assert len(team) == n_agents
        assert all([prob >= 0.0 and prob <= 1.0 for prob in team])
        shuffle(team)
        return team

    return get_distribution


register_distribution("fixed", fixed_attack_distribution)
