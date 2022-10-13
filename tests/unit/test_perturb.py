from unittest.mock import Mock
import numpy as np
import pytest
from smac.env.starcraft2.distributions import (
    ReflectPositionDistribution,
    SurroundedAndReflectPositionDistribution,
    SurroundedPositionDistribution,
    WeightedTeamsDistribution,
)


@pytest.mark.parametrize(
    ("selected_unit", "expected_ally_positions", "expected_enemy_positions"),
    [
        (0, np.array([[13, 20], [2, 3]]), np.array([[19, 20], [6, 7]])),
        (3, np.array([[0, 1], [13, 20]]), np.array([[4, 5], [19, 20]])),
    ],
)
def test_perturb_reflect(
    selected_unit, expected_ally_positions, expected_enemy_positions
):
    episode_config = {
        "ally_start_positions": {"item": np.array([[0, 1], [2, 3]]), "id": 0},
        "enemy_start_positions": {"item": np.array([[4, 5], [6, 7]]), "id": 0},
        "team_gen": {"item": ["nothing"], "id": 100},
    }
    capability_config = {
        "map_x": 32,
        "map_y": 32,
        "n_units": 2,
        "n_enemies": 2,
    }
    distribution = ReflectPositionDistribution(capability_config)
    new_rng = Mock()
    new_rng.integers.return_value = selected_unit
    distribution.rng = new_rng
    distribution.pos_generator = Mock()
    distribution.pos_generator.generate.return_value = {
        "ally_start_positions": {"item": np.array([[13, 20]])},
        "enemy_start_positions": {"item": np.array([[19, 20]])},
    }
    new_config = distribution.perturb(episode_config)
    expected_config = {
        "ally_start_positions": {
            "item": expected_ally_positions,
            "id": 0,
        },
        "enemy_start_positions": {"item": expected_enemy_positions, "id": 0},
        "team_gen": {"item": ["nothing"], "id": 100},
    }
    assert expected_config.keys() == new_config.keys()
    for env_key, env_key_dict in new_config.items():
        if env_key == "team_gen":
            assert env_key_dict == expected_config[env_key]
            continue
        assert (expected_config[env_key]["item"] == env_key_dict["item"]).all()


@pytest.mark.parametrize(
    (
        "uniform_return_values",
        "integer_return_values",
        "expected_enemy_positions",
    ),
    [
        ([0.3], [0, 0], np.array([[24, 24], [24, 24]])),
        ([0.7, 0.5], [0], np.array([[7, 7], [24, 24]])),
    ],
)
def test_perturb_surround(
    uniform_return_values, integer_return_values, expected_enemy_positions
):
    episode_config = {
        "ally_start_positions": {
            "item": np.array([[16, 16], [16, 16]]),
            "id": 0,
        },
        "enemy_start_positions": {
            "item": np.array([[4, 4], [24, 24]]),
            "id": 0,
        },
        "team_gen": {"item": "nothing", "id": 0},
    }
    capability_config = {
        "map_x": 32,
        "map_y": 32,
        "n_units": 2,
        "n_enemies": 2,
    }
    distribution = SurroundedPositionDistribution(capability_config)
    distribution.rng = Mock()
    distribution.rng.uniform.side_effect = uniform_return_values
    distribution.rng.integers.side_effect = integer_return_values

    perturbed_config = distribution.perturb(episode_config)
    expected_config = {
        "ally_start_positions": {
            "item": np.array([[16, 16], [16, 16]]),
            "id": 0,
        },
        "enemy_start_positions": {
            "item": expected_enemy_positions,
            "id": 0,
        },
        "team_gen": {"item": "nothing", "id": 0},
    }
    assert perturbed_config.keys() == expected_config.keys()
    for env_key, config in perturbed_config.items():
        if env_key == "team_gen":
            assert config == expected_config[env_key]
            continue
        assert (expected_config[env_key]["item"] == config["item"]).all()


@pytest.mark.parametrize(
    ("ally_positions", "is_reflect"),
    [
        (np.array([[16, 16], [16, 16]]), False),
        (np.array([[4, 4], [4, 18]]), True),
    ],
)
def test_perturb_surround_and_reflect(ally_positions, is_reflect):
    episode_config = {
        "ally_start_positions": {
            "item": ally_positions,
            "id": 0,
        },
        "enemy_start_positions": {
            "item": np.array([[4, 4], [24, 24]]),
            "id": 0,
        },
        "team_gen": {"item": "nothing", "id": 0},
    }
    capability_config = {
        "map_x": 32,
        "map_y": 32,
        "n_units": 2,
        "n_enemies": 2,
        "p": 0.5,
    }
    distribution = SurroundedAndReflectPositionDistribution(capability_config)
    distribution.surrounded_distribution = Mock()
    distribution.reflect_distribution = Mock()
    distribution.perturb(episode_config)
    if is_reflect:
        distribution.reflect_distribution.perturb.assert_called_once_with(
            episode_config
        )
    else:
        distribution.surrounded_distribution.perturb.assert_called_once_with(
            episode_config
        )


@pytest.mark.parametrize(
    (
        "ally_team",
        "enemy_team",
        "unit_to_change",
        "new_unit_type",
        "expected_ally_team",
        "expected_enemy_team",
    ),
    [
        (
            ["stalker", "stalker", "zealot"],
            ["stalker", "stalker", "zealot"],
            1,
            "zealot",
            ["stalker", "zealot", "zealot"],
            ["stalker", "zealot", "zealot"],
        ),
        (
            ["stalker", "stalker", "zealot"],
            ["stalker", "stalker", "zealot"],
            3,
            "zealot",
            ["zealot", "stalker", "zealot"],
            ["zealot", "stalker", "zealot"],
        ),
        (
            ["stalker", "stalker", "zealot"],
            ["stalker", "stalker", "zealot", "colossus"],
            6,
            "stalker",
            ["stalker", "stalker", "zealot"],
            ["stalker", "stalker", "zealot", "stalker"],
        ),
    ],
)
def test_unit_type_perturbation(
    ally_team,
    enemy_team,
    unit_to_change,
    new_unit_type,
    expected_ally_team,
    expected_enemy_team,
):
    episode_config = {
        "teams": {
            "ally_team": ally_team,
            "enemy_team": enemy_team,
            "id": 0,
        },
        "start_positions": {"item": "nothing", "id": 0},
    }

    capability_config = {
        "env_key": "teams",
        "n_units": len(ally_team),
        "n_enemies": len(enemy_team),
        "weights": [0.45, 0.45, 0.1],
        "unit_types": ["stalker", "zealot", "colossus"],
        "exception_unit_types": ["colossus"],
    }
    distribution = WeightedTeamsDistribution(capability_config)
    distribution.rng = Mock()
    distribution.rng.integers.return_value = unit_to_change
    distribution.rng.choice.return_value = new_unit_type
    new_episode_config = distribution.perturb(episode_config)
    expected_config = {
        "teams": {
            "ally_team": expected_ally_team,
            "enemy_team": expected_enemy_team,
            "id": 0,
        },
        "start_positions": {"item": "nothing", "id": 0},
    }
    assert expected_config == new_episode_config
