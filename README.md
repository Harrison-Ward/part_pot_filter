# Particle Filter Estimation for Auction Pot Size

## Overview

This repository contains a Python implementation of a particle filter designed to estimate the pot size in an auction setting, specifically tailored for situations where bids reflect the underlying value of items being auctioned (e.g., teams in a sports betting scenario) and the total pot size that accumulates from these bids. A particle filter is a powerful tool for sequential estimation that can handle non-linear models and non-Gaussian noise, making it ideal for the complex dynamics typically observed in auctions.

## Particle Filter Model

### Latent Transition and Update Mechanism

The latent transition of the particles is modeled using a Gaussian distribution to simulate the evolution of each particle's state. For each iteration, we predict the next state based on a normal distribution centered around the current state with a standard deviation reflecting the process noise:

\[ particles_{t+1} = particles_t + \mathcal{N}(0, \sigma_{process}) \]

The update step incorporates observed bids to adjust the weights of the particles. The likelihood of observing a bid given a particle's state is modeled with a normal distribution centered around the predicted bid (\(particle * win\_probability\)) with a standard deviation representing the measurement noise:

\[ \text{likelihood} = \mathcal{N}(observed\_bid; particle * win\_probability, \sigma_{measurement}) \]

### Resampling Strategy

Systematic resampling is employed to address the degeneracy problem, ensuring that particles with higher weights are more likely to be selected for the next generation. This method contributes to focusing computational resources on the most probable states.

## Simulation Strategy

The model is tested through a simulation that generates observed bids based on a true pot value and a defined win probability. Observed bids are simulated using a normal distribution centered around the true bid value (\(true\_pot\_value * win\_probability\)) with variability introduced by a specified standard deviation. The particle filter processes each bid sequentially, updating estimates of the pot size and refining the particles' distribution to converge towards the true pot value.

## Visualizations

To illustrate the performance and results of the particle filter, two key visualizations are generated at the bottom of the script:

1. **Time Series Estimation**: This graph shows the estimated pot size as a function of observed bids over time, comparing it to the true pot value.

   ![Estimation of Pot Size as a Function of Bids Observed](./outputs/estimate_time_series.png)

2. **Kernel Density Estimate (KDE)**: A KDE plot visualizes the final distribution of particle states, with a vertical line marking the true pot size, showcasing the model's accuracy and uncertainty.

   ![Kernel Density Estimate of Final Particle States](./outputs/particle_kde.png)

These visualizations provide insight into the model's dynamics, demonstrating its ability to adapt and refine the pot size estimate with each new bid observed.