from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup

description = """SMACv2 - StarCraft Multi-Agent Challenge

SMACv2 is an update to Whirl’s Starcraft Multi-Agent Challenge, 
which is a benchmark for research in the field of cooperative 
multi-agent reinforcement learning. SMAC and SMACv2 both focus 
on decentralised micromanagement scenarios in StarCraft II, 
rather than the full game.

The accompanying paper which outlines the motivation for using SMAC as well as
results using the state-of-the-art deep multi-agent reinforcement learning
algorithms can be found at https://www.arxiv.link

Read the README at https://github.com/oxwhirl/smacv2 for more information.
"""

extras_deps = {
    "dev": [
        "pre-commit>=2.0.1",
        "black>=19.10b0",
        "flake8>=3.7",
        "flake8-bugbear>=20.1",
    ],
}


setup(
    name="SMACv2",
    version="1.0.0",
    description="SMACv2 - StarCraft Multi-Agent Challenge.",
    long_description=description,
    author="WhiRL",
    author_email="benellis@robots.ox.ac.uk",
    license="MIT License",
    keywords="StarCraft, Multi-Agent Reinforcement Learning",
    url="https://github.com/oxwhirl/smacv2",
    packages=[
        "smacv2",
        "smacv2.env",
        "smacv2.env.starcraft2",
        "smacv2.env.starcraft2.maps",
        "smacv2.env.pettingzoo",
        "smacv2.bin",
        "smacv2.examples",
        "smacv2.examples.rllib",
        "smacv2.examples.pettingzoo",
    ],
    extras_require=extras_deps,
    install_requires=[
        "pysc2>=3.0.0",
        "protobuf<3.21",
        "s2clientprotocol>=4.10.1.75800.0",
        "absl-py>=0.1.0",
        "numpy>=1.10",
        "pygame>=2.0.0",
    ],
)
