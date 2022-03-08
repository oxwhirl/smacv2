from typing import Set, List, Optional
from itertools import combinations_with_replacement
from random import choice, shuffle

DISTRIBUTIONS = {}


def get_distribution_function(teammate_distribution: Optional[str]):
    return DISTRIBUTIONS[teammate_distribution]


def register_distribution(distribution_str: str, distribution_fn):
    DISTRIBUTIONS[distribution_str] = distribution_fn


def stub_distribution_function(
    n_ally_units: int,
    **kwargs,
):
    """Stub distribution function used when the meta_marl setting of SMAC is
    turned off. Raises a NotImplementedError.

    Args:
        ally_unit_types (List[str]): A list of unit types available to the ally team
        n_ally_units (int): The number of ally units
        enemy_units (List[object]): The list of enemy units that will be faced. This can
            be used to generate a team that matches the enemies' capabilities.
    """

    def get_team(test_mode=False):
        raise NotImplementedError("No distribution of teammates specified!")

    return get_team, {}


register_distribution("stub", stub_distribution_function)


def all_teams_distribution_function(
    n_ally_units: int,
    **kwargs,
):
    """Distribution function that just cycles through all possible ally team
    distributions. Args as specified in `stub_distribution_function`
    """
    all_combinations = list(
        combinations_with_replacement(kwargs["ally_unit_types"], n_ally_units)
    )

    def get_team(test_mode=False):
        while True:
            team = list(choice(all_combinations))
            team_id = all_combinations.index(tuple(team))
            shuffle(team)
            yield team, team_id

    return get_team, {
        "n_train_tasks": len(all_combinations),
        "n_test_tasks": len(all_combinations),
    }


register_distribution("all", all_teams_distribution_function)


def fixed_team_distribution_function(
    n_ally_units: int,
    **kwargs,
):
    train_team_list = list(kwargs["ally_train_team_compositions"])
    test_team_list = list(kwargs["ally_test_team_compositions"])

    def get_team(test_mode=False):
        if not test_mode:
            while True:
                team = list(choice(train_team_list))
                team_id = train_team_list.index(team)
                assert len(team) == n_ally_units
                # assert all(
                #     [
                #         team_member_type in ally_unit_types
                #         for team_member_type in team
                #     ]
                # )
                shuffle(team)
                yield team, team_id
        else:
            while True:
                for team_id, team in enumerate(test_team_list):
                    assert len(team) == n_ally_units
                    # assert all(
                    #     [
                    #         team_member_type in ally_unit_types
                    #         for team_member_type in team
                    #     ]
                    # )
                    shuffle(team)
                    yield team, team_id

    return get_team, {
        "n_train_tasks": len(train_team_list),
        "n_test_tasks": len(test_team_list),
    }


register_distribution("fixed_team", fixed_team_distribution_function)
