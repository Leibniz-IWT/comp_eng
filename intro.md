---
title: Computational Methods – WiSe 2025/2026
---

# Computational Methods – WiSe 2025/2026

## Introduction

Welcome! In this module you will learn a wide set of methods in the domain of computational engineering which will aid you in establishing practical skills in your engineering career.

This course bridges many topics which are traditionally taught separately in order to establish a cohesive framework which connects everything else that you have learned and/or will learn throughout the course of your graduate studies back to _a_ practical framework that you are comfortable with using. The focus of this course is on what is inherently computational (and just as importantly what _isn't_ computational!), with theoretical depth limited to what is useful for practicing engineers to know in their everyday work. At the end of each unit further literature is suggested for more in-depth study. This course is intentionally structured in a such a way that concepts introduced are always given with code that can be played with, this is essential for learning these skillsets and establishing a practical working knowledge of all the methods introduced here. While the context and examples are focussed on space engineering, these methods are, of course, used in a wide variety of engineering disciplines.

Below you’ll find links to each unit's notebook. Each unit is meant to be studied in a single week.

## Syllabus

| Unit # | Title & key sub-topics |
|-----------|-----------------------|
| **1** | **{doc}`Model classification I – building blocks <notebooks/01_model-classification>`**<br>• Model classification and structure: linear / non-linear, coupled / uncoupled, symmetry, static / dynamic<br>• Conservation rules & force balances |
| **2** | **{doc}`Model classification II – degrees of freedom <notebooks/02_dof-analysis>`**<br>• DOF analysis & constraints<br>• Determined vs. under-/over-determined problems |
| **3** | **{doc}`Model classification III – non-linearity & chaos <notebooks/03_chaos>`**<br>• Fixed-, limit-cycle- & chaotic responses (logistic map, Feigenbaum)<br>• Continuum mechanics as “infinite DOF” ⇒ discretisation need |
| **4** | **{doc}`Simulation I – algebraic systems <notebooks/04_simulation-algebraic>`**<br>• Linear solves (`scipy.linalg.solve`), conditioning & scaling<br>• Non-linear solves (`fsolve`), Jacobians (`sympy`), Hessians & Newton |
| **5** | **{doc}`Simulation II – ODE / DAE systems <notebooks/05_simulation-ode-dae>`**<br>• Explicit vs. implicit, stiffness, `solve_ivp`<br>• Index-1 differential-algebraic equations |
| **6** | **{doc}`Simulation III – continuum / PDE intro <notebooks/06_simulation-pde>`**<br>• Method-of-lines; 1-D heat eqn<br>• Coupled PDE systems |
| **7** | **{doc}`Optimisation I – convex & linear <notebooks/07_optimisation-linear>`**<br>• LP / QP (`scipy.optimize.linprog`)<br>• Linear least-squares<br>• Gradient-descent methods |
| **8** | **{doc}`Optimisation II – non-linear programming <notebooks/08_optimisation-nonlinear>`**<br>• KKT, trust-region & SQP (`minimize`)<br>• Non-linear least-squares, back-prop, SGD |
| **9** | **{doc}`Optimisation III – global / derivative-free <notebooks/09_optimisation-global>`**<br>• Derivative-free & black-box methods<br>• Bayesian optimisation, integer & discontinuous variables |
| **10** | **{doc}`Control I – classical feedback <notebooks/10_control-classical>`**<br>• Loop anatomy, stability margins<br>• P/PI/PD/PID tuning (root-locus & frequency domain) |
| **11** | **{doc}`Control II – state-space & optimal linear <notebooks/11_control-lqr>`**<br>• LTI form, controllability / observability<br>• LQR, discrete Riccati, Kalman filter, LQG |
| **12** | **{doc}`Control III – model predictive & non-linear control <notebooks/12_control-mpc>`**<br>• Linear & nonlinear MPC, warm-start<br>• Successive linearisation, constraint handling<br>• Intro to SINDy / DMDc for data-driven MPC |
| **13** | **{doc}`Control IV – reinforcement-learning foundations <notebooks/13_control-rl-foundations>`**<br>• MDPs, value & policy functions<br>• Tabular & deep Q-learning, actor-critic<br>• Safety envelopes & reward shaping |
| **14** | **{doc}`RL-based orbit station-keeping <notebooks/14_rl-station-keeping>`**<br>• Problem statement & environment definition<br>• Agent training, evaluation, fuel trade-off<br>• Frontiers: hierarchical RL, onboard edge compute |

## Prerequisites

Students for this course are expected to have a foundation undergraduate engineering topics especially in calculus, linear algebra, differential equations and dynamics. While a background in more senior topics such as control, fluid mechanics and machine learning is beneficial, they are not essential since these topics are re-introduced and therefore self-contained for the purposes of completing this course. However, since only the computational aspects of these topics are covered, it is highly recommended for students to pursue external sources for a more in depth education on these topics. Basic python programming skills are an essential prequisite.

## Course information

Instructor contact information:

TAs: TBD

## Grading policy 

TBD

## Help with setting up your system, python environments and running these notebooks offline.

Lorem ipsum






```{contents}
:local:
:depth: 2
