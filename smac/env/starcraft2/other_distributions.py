from re import I
import numpy as np
from random import choice, shuffle

DISTRIBUTIONS = {}


def get_distribution(distribution_key):
    return DISTRIBUTIONS[distribution_key]


def register_distribution(key, fn):
    DISTRIBUTIONS[key] = fn


def stub_distribution_function(n_agents, **kwargs):
    def distribution(test_mode=False):
        while True:
            yield [0.0] * n_agents, 0

    return distribution


register_distribution("stub", stub_distribution_function)


def uniform_distribution(n_agents, **kwargs):
    rng = np.random.default_rng()

    def distribution(test_mode=False):
        while True:
            yield rng.uniform(
                low=kwargs["attack_probability_low"],
                high=kwargs["attack_probability_high"],
                size=(n_agents,),
            ), 0

    return distribution


register_distribution("uniform", uniform_distribution)


def fixed_attack_distribution(n_agents, **kwargs):
    train_distributions = list(kwargs["train_distributions"])
    test_distributions = list(kwargs["test_distributions"])

    def get_distribution(test_mode=False):
        if not test_mode:
            while True:
                team = list(choice(train_distributions))
                team_id = train_distributions.index(team)
                assert len(team) == n_agents
                assert all([prob >= 0.0 and prob <= 1.0 for prob in team])
                shuffle(team)
                yield team, team_id
        else:
            while True:
                for team_id, team in enumerate(test_distributions):
                    assert len(team) == n_agents
                    assert all([prob >= 0.0 and prob <= 1.0 for prob in team])
                    shuffle(team)
                    yield team, team_id

    return get_distribution


register_distribution("fixed", fixed_attack_distribution)
