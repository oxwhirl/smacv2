from random import choice
from smac.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper


class AdversarialSC2Env(StarCraftCapabilityEnvWrapper):
    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def reset_random(self):
        """Resets to a random level"""
        return super().reset()

    def reset_agent(self):
        """Resets the current level to the start state without changing
        the capability config"""
        return self.env.reset(episode_config=self.env.episode_config)

    def reset_to_level(self, level):
        """Resets the environment to the specific config given
        by level"""
        return self.env.reset(episode_config=level)

    def reset(self):
        """Supposed to reset the env to a blank state so it can be
        designed by an agent. However, can just use reset_random instead"""
        raise NotImplementedError("Reset not a valid method for this Env")

    def mutate_level(self, num_edits):
        """Randomly edit either the start position or the
        unit type and then reset this level."""
        episode_config = self.env.episode_config
        for _ in range(num_edits):
            # choose a distribution to perturb
            distributions = list(
                set(self.env_key_to_distribution_map.values())
            )
            distribution = choice(distributions)
            episode_config = distribution.perturb(episode_config)
        return self.reset_to_level(episode_config)
