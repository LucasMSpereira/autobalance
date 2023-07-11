# autobalance

A game is considered to be "balanced" when, in similar contexts of unconstrained choice, the players don’t necessarily make the same decision. Examples:

<ul>
  <li>In CSGO, in the beginning of a round, if the player has maximum money and faces a similar situation many times, they will not buy the same weapon every time.</li>
  <li>In Dota 2, if a support player is going to make the first pick, they won’t always pick the same hero, but rather choose from a reasonably sized subgroup of heroes.</li>
</ul>

Decision variety by good players in repetition of similar situations indicates that no particular choice will considerably increase the chances of winning (or losing). In contrast, constancy of choice in these contexts would indicate an unbalanced element of the game (a.k.a. “overpowered” element). When this definition of balance is achieved, in these situations, players can make choices based on their subjective preference. Furthermore, this will not have to be done by sacrificing their chances of winning (i.e., choosing between having fun or winning).

## Overview

<p align = "center">
  <img src = "https://github.com/LucasMSpereira/autobalance/assets/84910559/9645b84c-d088-47cc-a65f-82f35e23e737">
</p>

This project aims at automating the process of game balancing. A game engine defines a parametrized environment. Then, reinforcement learning agents are trained on the current version of this environment to simulate good players. When the run ends, statistics are collected, describing element pick and success rate tendencies.

The data is then used in bayesian optimization to update the parameters of the game environment. After successful optimization, the statistics should indicate a uniform pick and win-rate across game elements.
