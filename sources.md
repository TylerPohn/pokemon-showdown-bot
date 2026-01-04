Reinforcement Learning Resources for Pokémon Showdown (OU Format)
Datasets of Pokémon Showdown OU Battles
PokéChamp Replay Dataset (Gen1–9 OU and more) – A large public dataset of ~2 million competitive battle replays across 37+ formats (including all OU tiers) from 2024–2025
pokeagent.github.io
. The data (hosted on Hugging Face) can be filtered by format (e.g. Gen9 OU) for supervised or offline RL training
github.com
github.com
. This “PokéChamp” dataset is part of an ICML 2025 project and provides cleaned battle logs (with train/test splits) covering a wide Elo range
github.com
.
Metamon Replays (OU Trajectories) – Another extensive dataset from the Metamon project (UT Austin, 2023), focusing on offline RL. It includes ~1.8M raw battle logs (2014–2025)
pokeagent.github.io
 and over 3.5M parsed battle trajectories for RL
pokeagent.github.io
. The parsed version reconstructs each battle into state-action sequences, making it easier to train agents with imperfect information. Data is organized by format (e.g. separate files for gen9ou, gen8ou, etc.)
huggingface.co
 – each contains many OU battles with turn-by-turn JSON state data
huggingface.co
. (Code is available via the Metamon repo for loading these trajectories.)
Kaggle and Other Community Logs – The community has also shared battle logs. For example, a Kaggle dataset of 13,000+ parsed Pokémon Showdown battles (Gen9 Random Battle format) was published in 2025. While that set is Random Battles, it demonstrates methods to collect and parse replays. Similar approaches can be applied to OU – e.g. using Showdown’s public replay API to gather Gen9 OU games. In fact, the Showdown replay site allows filtering by format (like ?format=gen9ou), and appending .json to a replay URL returns a JSON with the full battle log
smogon.com
smogon.com
. This makes it feasible to scrape large OU datasets for training if an official dataset is not readily available.
Frameworks and Libraries for Pokémon Showdown AI
poke-env (Python) – A widely used Python library for building RL agents on Pokémon Showdown. It provides an easy API to handle Pokémon battles and integrates with OpenAI Gym/Gymnasium for RL training
ivison.id.au
. Essentially, poke-env wraps the Showdown battle simulator so you can plug in neural network policies (e.g. with Stable-Baselines3 or custom algorithms)
ivison.id.au
. It manages game state objects (Pokémon, moves, etc.) and can connect to either a local or online Showdown server for battles
poke-env.readthedocs.io
. This library is actively maintained and is the backbone of many recent academic projects – it’s even recommended as the go-to interface “used by most recent academic work” on Showdown AI
pokeagent.github.io
. The documentation includes examples for creating custom agents, reward functions, and running training loops (e.g. DQN or PPO) in an OU format environment.
PokéChamp & Metamon Codebases – Both the PokéChamp (ICML 2025) and Metamon (NeurIPS 2025) projects have open-source repositories demonstrating advanced Showdown AI frameworks. These are research-level frameworks, but they support training agents with neural networks:
PokéChamp implements an OU battle agent using a minimax-inspired Large Language Model approach (with battle text prompts), and also provides tools for supervised move prediction and team prediction
github.com
github.com
. Its repo includes a battle environment and evaluation scripts for Gen9 OU
github.com
.
Metamon focuses on offline RL: it trained high-performing agents on human OU replays (reportedly reaching top 10% on the ladder from 475k+ training games). The Metamon code (UT-Austin-RPL/metamon) includes data loaders for replay trajectories
huggingface.co
, model definitions (Transformer-based policies), and evaluation against baseline bots. These frameworks are more complex but show how to train OU agents from logged data at scale.
Other Bot Frameworks – Several community projects provide Showdown bot skeletons and simulators:
Taylor Hansen’s Showdown AI (Python+TS) – A reinforcement learning framework that includes a Pokémon Showdown bot client (PsBot), a battle state parser, and a neural network training pipeline
github.com
. It was designed for Gen4 random battles (self-play training ~16k games) but could be adapted. It connects to a Showdown server (local or live) and lets your model play games via the Showdown protocol
github.com
.
“Leftovers Again” and others – Older projects like leftovers-again (a bot client platform) and shallow-red (a bot using Monte Carlo Tree Search) exist on GitHub. While not focused on modern neural networks, they might be useful for ideas on state representation or baseline heuristics. However, for RL with neural nets in Python, poke-env remains the most up-to-date and OU-ready solution.
Example Projects and Tutorials
Tutorial – Training a Showdown RL Agent (poke-env) – A detailed Medium tutorial by Vivek Keval et al. walks through building a Gen9 Random Battle agent using poke-env and stable-baselines3
medium.com
medium.com
. It covers setting up a local Showdown server, installing poke-env, and writing a custom Gen9EnvSinglePlayer agent class with a neural network policy. The guide demonstrates how to encode battle state (HP, moves, types, etc.) into a numeric observation and train an RL model (DQN) against simple bots. This tutorial’s code can be adapted to OU battles by specifying battle_format="gen9ou" in the environment and using real teams.
Hamish Ivison’s Blog – RL for Pokémon Battles – An insightful blog post where the author trains a DQN agent (with a neural network) to play Pokémon battles using poke-env
ivison.id.au
. He explains how to convert the game state into features and reward functions for effective learning. The project uses Stable-Baselines3 with poke-env’s OpenAI Gym interface, training the agent against built-in baseline opponents (like Random moves or Max Damage). It provides code snippets for the custom SimpleRLPlayer and demonstrates evaluation results in a Gen8 environment
ivison.id.au
ivison.id.au
. This is a great starting point for understanding how to glue together the environment, model, and training loop for Showdown AI.
Pokémon Showdown AI on Medium (Viet Nguyen) – A Medium article by Viet Nguyen (2023) documents a project to build a Showdown battle AI. It starts by collecting a large dataset of battle logs and constructing features for each turn’s game state
medium.com
. The author created a supervised learning model (TensorFlow) to predict the probability of winning from any given state, as a step toward a full battling AI. The article (and accompanying GitHub repo) shows how the raw replay logs can be processed into a tabular dataset with hundreds of features (Pokémon, HP, status, field conditions, etc.)
medium.com
. While this project focuses on win prediction and uses AWS SageMaker for training, it provides a useful example of dataset preparation and a neural network applied to Showdown data.
GitHub Repos – Competitive Battle AIs: There are several open-source repos of interest:
hsahovic/poke-env – (Mentioned above) includes example agents and a Gym interface for Showdown battles
ivison.id.au
.
sethkarten/pokechamp – Code for the PokéChamp OU agent, including move prediction models and a battle engine. It features a Bayesian move/item prediction module trained on replays
github.com
github.com
, and scripts to simulate battles between agents or on the ladder.
UT-Austin-RPL/metamon – Code for the Metamon offline RL framework, showing how to train transformer-based policies on logged OU battles. It provides utilities to load the parsed trajectories and train/evaluate agents in a simulated environment.
Eresia/Pokemon-Showdown-AI – An older project (2016) that implemented a Showdown bot with heuristic strategies. Though outdated, it may offer insights into the structure of a Showdown AI client.
In summary, Pokémon Showdown (OU) has a rich set of resources for AI development. Large battle datasets are publicly available for offline training (e.g. via PokéChamp or by scraping replays)
pokeagent.github.io
smogon.com
. For frameworks, the community standard is to use poke-env in Python, which provides a battle environment for RL agents
ivison.id.au
. On top of that, cutting-edge projects like PokéChamp and Metamon have open-sourced their code and data, demonstrating high-level approaches to mastering OU battles. By leveraging these datasets and tools, you can train a neural network model (via supervised learning or reinforcement learning) to play Pokémon Showdown in OU – following in the footsteps of prior bots and research prototypes
pokeagent.github.io
medium.com
. Sources:
Showdown OU replay datasets and API usage
pokeagent.github.io
smogon.com
PokéChamp and Metamon project documentation
github.com
pokeagent.github.io
poke-env library docs and usage examples
ivison.id.au
poke-env.readthedocs.io
Community tutorials and project write-ups
medium.com
medium.com
GitHub repositories for Showdown AI bots and frameworks
github.com
pokeagent.github.io
Citations
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html

jakegrigsby/metamon-parsed-replays · Datasets at Hugging Face

https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays

jakegrigsby/metamon-parsed-replays · Datasets at Hugging Face

https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays
Need help gathering and parsing battle logs | Smogon Forums

https://www.smogon.com/forums/threads/need-help-gathering-and-parsing-battle-logs.3767924/
Need help gathering and parsing battle logs | Smogon Forums

https://www.smogon.com/forums/threads/need-help-gathering-and-parsing-battle-logs.3767924/
Reinforcement Learning with Pokemon | Hamish Ivison

https://ivison.id.au/2021/08/02/pokerl.html

Poke-env: A Python Interface for Training Reinforcement Learning Pokémon Bots — Poke-env documentation

https://poke-env.readthedocs.io/en/stable/
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp

jakegrigsby/metamon-parsed-replays · Datasets at Hugging Face

https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays

GitHub - taylorhansen/pokemonshowdown-ai: Reinforcement learning for Pokemon.

https://github.com/taylorhansen/pokemonshowdown-ai

GitHub - taylorhansen/pokemonshowdown-ai: Reinforcement learning for Pokemon.

https://github.com/taylorhansen/pokemonshowdown-ai

Pokémon Showdown with Reinforcement Learning | by Vivek Keval | Medium

https://medium.com/@kevalvivek4/pok%C3%A9mon-showdown-with-reinforcement-learning-f5aad9a93cef

Pokémon Showdown with Reinforcement Learning | by Vivek Keval | Medium

https://medium.com/@kevalvivek4/pok%C3%A9mon-showdown-with-reinforcement-learning-f5aad9a93cef
Reinforcement Learning with Pokemon | Hamish Ivison

https://ivison.id.au/2021/08/02/pokerl.html
Reinforcement Learning with Pokemon | Hamish Ivison

https://ivison.id.au/2021/08/02/pokerl.html

A Naive Pokemon Showdown TensorFlow Model for Winning Prediction with AWS Sagemaker | by Viet Nguyen | Medium

https://medium.com/@nguyenhongviet1997/a-naive-pokemon-showdown-tensorflow-model-for-winning-prediction-with-aws-sagemaker-260f59915738
Reinforcement Learning with Pokemon | Hamish Ivison

https://ivison.id.au/2021/08/02/pokerl.html

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp

GitHub - sethkarten/pokechamp: Official repository of the spotlight ICML 2025 paper, PokeChamp: an Expert-level Minimax Language Agent.

https://github.com/sethkarten/pokechamp
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html
Track 1: Competitive Battling - PokéAgent Challenge

https://pokeagent.github.io/track1.html
