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

---
## Help with setting up your system, python environments and running these notebooks offline.

### Running the notebooks

You can run the notebooks in two ways: (1) Binder (fully online, no install needed) or (2) local install with full control.

#### 1. Launch instantly on Binder  🚀  

Click the **“Launch Binder”** badge at the top-right of any page (or the button below).  
Binder will build a temporary Jupyter-Lab session in the cloud; nothing to install on your computer.

> **Note:** The build takes a couple of minutes, a read-only preview appears first.  
> Leave the tab open: your interactive session will load automatically when ready.

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Leibniz-IWT/comp_eng/main?urlpath=lab/tree/notebooks/01_model-classification.ipynb)

---
#### 2. Run locally on your system with conda
##### macOS / Linux — Terminal and Windows — PowerShell

1. **Clone the repository and set up a local conda environment (once-off)**:

   ```bash
    git clone https://github.com/Leibniz-IWT/comp_eng.git
    cd comp_eng
    conda env create -f environment.yml     # creates “compeng”
    conda activate compeng
    ```
You can now start either jupyter lab or classic notebook or lab

2. **Daily work flow (every time you start a new terminal session)**:
    ```bash
    conda activate compeng  # Reactive the environment
    jupyter lab          # or: jupyter notebook
    ```
Your browser will open at `http://localhost:8888`, then launch **notebooks/01_model-classification.ipynb**.

4. (Optional) **Updating later:**
Pull the repo (git pull) and refresh the environment (conda env update -f environment.yml).

   ```bash
   $ git pull
   $ conda env update -f environment.yml

> **Note:** For every new terminal session you need reactive the environment:  
> `conda activate compeng` _before_ running Jupyter or you could run into trouble when your system conda reverts to the base environment

---

### Windows — PowerShell / Cmd / Git Bash

In Windows, after installing git, you can either use powershell and follow the instructions above or the GitHub Desktop app:

_In the GitHub Desktop app window:_

1. *File → Clone Repository…* → choose **comp_eng**.  
2. *Repository → Open in Terminal* (or *Open in Git Bash*).  
3. Run the same commands shown above.

---

## Updating later (all platforms)
```bash
cd comp_eng
git pull                               # fetch latest notebooks/code
conda activate compeng
conda env update -f environment.yml    # refresh dependencies
```

Choose whichever option fits your workflow and personal taste: Binder for pure cloud environment, conda for local/off-line work.

---
## Nomenclature

In this course, we will use the following notation for variables, vectors, matrices, and functions except where otherwise stated. This is a common notation in engineering and scientific computing, and it is important to be familiar with it as you progress through the course.

| Notation | Interpretation                                                                                 |
|----------|------------------------------------------------------------------------------------------------|
| $x_i$ | Scalar _static_ variable                                                                       |
| $\mathbf{x}$ | Column vector of static variables, $\; \mathbf{x}=[x_1,\dots,x_n]^\top$                        |
| $\mathbf{X}$ | Matrix (rank-2 array) of static quantities, including scalar fields                            |
| $\mathcal{X}$ | Set or domain in which $\mathbf{x}$ lives (e.g.\ feasible design set)                          |
| $y_i(t)$, $y_i$ | Scalar _dynamic_ variable (time-dependent); if $t$ is omitted, instantaneous value is implied  |
| $\mathbf{y}(t)$ | State vector of dynamic variables, $\; \mathbf{y}=[y_1,\dots,y_m]^\top$                        |
| $\mathbf{Y}$ | Matrix assembled from vectors $\mathbf{y}(t)$ (e.g.\ trajectory snapshots)                     |
| $\mathcal{Y}$ | Set of admissible trajectories or outputs                                                      |
| $a_i$ | Scalar coefficient (always known fixed values)                                                 |
| $\mathbf{a}$ | Vector of coefficients                                                                         |
| $\mathbf{A}$ | Coefficient matrix in a linear system $\mathbf{A}\mathbf{x}=\mathbf{b}$                        |
| $p_i$ | Scalar _parameter_ (fixed but possibly uncertain, e.g. found by solving optimization problems) |
| $\mathbf{p}$ | Parameter vector                                                                               |
| $\mathbf{P}$ | Parameter matrix (if parameters are organised in matrix form)                                  |
| $f(\cdot)$ | Scalar-valued function (mapping $f: \mathbb{R}^n\!\to\!\mathbb{R}$)                            |
| $\mathbf{f}(\cdot)$ | Vector-valued mapping (e.g.\ residual $\mathbf{f}:\mathbb{R}^n\!\to\!\mathbb{R}^n$)            |



```{contents}
:local:
:depth: 2
