# Reproducing SMACv2 Experiments

Logging of the experiments is via [wandb.ai](https://wandb.ai/). Before trying to replicate the experiments here, you will have to create a WandB account. If you have any questions about these instructions please [raise an issue](https://github.com/oxwhirl/smacv2/issues/new/choose).

To make WandB work, you will need to copy your wandb api key. You can do this by going to your image on the top right > user settings > danger zone > API keys. Copy this and put it into a file. Set the location of this file in an environment variable called `WANDB_API_KEY_FILE`. Use, for example, `export WANDB_API_KEY_FILE=$HOME/.wandb_api_key`. 

# Running SMACv2 Baselines

## QMIX

0. Make sure you have correctly set the WANDB_API_KEY_FILE environment variable mentioned in the introduction.
1. Clone the pymarl2 repository:
```git clone https://github.com/benellis3/pymarl2.git```
2. Build the Docker container by running `docker build -t pymarl2:ben_smac -f docker/Dockerfile --build-arg UID=$UID .` from the pymarl2 directory.
3. Install Starcraft by running `./install_sc2.sh` in the pymarl2 directory.
4. Navigate to `src/config/default.yaml` and set `project` and `entity` to your desired project name (can be anything) and your wandb username respectively.
5. Set `td_lambdas` in `run_exp.sh` (line 21) to be `0.4` and `eps_anneal` to `100000`.
6. Run `./run_exp.sh qmix <tag>` where `<tag>` is a word to help you identify the experiments. If you want to run the `10_vs_11` or `20_vs_23` scenarios, you will have to ensure the `./run_docker.sh` command on line 46 has `n_units` and `n_enemies` set correctly. For example for the `20_vs_23` scenario you would set `n_units=20` and `n_enemies=23`.

## MAPPO

0. Make sure you have correctly set the WANDB_API_KEY_FILE environment variable mentioned in the introduction.
1. Clone the MAPPO repository:
```git clone https://github.com/benellis3/mappo.git```
2. Build the docker container by running `build.sh` in the `docker directory`
3. Install Starcraft by running `./install_sc2.sh` in the mappo directory.
4. Navigate to `src/config/default.yaml` and set `project` and `entity` to your desired project name (can be anything) and your wandb username respectively.
5. Set `lr` in `run.sh` to `0.0005` and `clip_range` to `0.1`. If you want to run the closed-loop baseline, change `maps` to only contain maps *without* `open_loop` in their name. For the open-loop baseline, do the opposite, i.e. keep all the map names with `open_loop` in them and delete the rest.
6. Run `./run.sh clipping_rnn_central_V <tag>` where `<tag>` is a word to help you identify the experiments. If you want to run the `10_vs_11` or `20_vs_23` scenarios, you will have to set `offset` on line 21 of the script. This controls how many more enemies there are than allies.

# Running EPO Baselines

## QMIX

1. Complete steps 0-4 of Running SMACv2 Baselines (QMIX), making sure to use `run_exp_epo.sh` where `run_exp.sh` is mentioned.
2. Run `./run_exp_epo.sh qmix <tag>` where `<tag>` is a word to help you identify the experiments.

## MAPPO

1. Complete steps 0-4 of Running SMACv2 Baselines (MAPPO), making sure to use `run_exp_epo.sh`. where `run_exp.sh` is mentioned.
2. Run `./run_exp_epo.sh mappo <tag>` where `<tag>` is a word to help you identify the experiments.

# Running Open-Loop SMAC baselines

## MAPPO

1. Follow steps 0 and 1 of Running SMACv2 Baselines (MAPPO).
2. Checkout the `stochastic-experiment` branch:
   ```git checkout stochastic-experiment```
3. Install Starcraft by running `./install_sc2.sh` in the mappo directory.
4. Build the docker container by running `./build.sh` in the `docker` directory.
5. Navigate to `src/config/default.yaml` and set `project` and `entity` to your desired project name (can be anything) and your wandb username respectively.
6. Run `./run_exp.sh clipping_rnn_central_V <tag>` where `<tag>` is a word to help you identify experiments.

## QMIX

1. Follow steps 0 and 1 of Running SMACv2 Baselines (QMIX).
2. Checkout the `stochastic_test` branch:
```git checkout stochastic_test```
3. Install Starcraft by running `./install_sc2.sh` in the pymarl2 directory.
4. Build the Docker container by running `./build.sh` in the 
`docker` directory.
6. Navigate to `src/config/default.yaml` and set `project` and `entity` to your desired project name (can be anything) and your wandb username respectively.
7. Run `./run.sh qmix <tag>` where `<tag>` is a word to help you identify experiments.

# Running Q-Regression Experiments

See the README in the [pymarl2 repo](https://github.com/benellis3/pymarl2/tree/smacv2-feature-inferrability)
