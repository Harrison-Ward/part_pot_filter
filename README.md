# Bayesian Estimation of Total Pot Size During Live Auctions

## Overview

This is a Python implementation of a particle filter designed to estimate the pot size in an auction setting, specifically tailored for situations where bids reflect the underlying value of items being auctioned (e.g., teams in a sports betting scenario) and the total pot size that accumulates from these bids. A particle filter is a powerful tool for sequential estimation that can handle non-linear models and non-Gaussian noise, making it ideal for the complex dynamics typically observed in auctions.

To add ensure that this approach is sufficiently robust, the particle filter's estimates are averaged with a simple model that assumes the fraction of the pot observed so far is proportional to the total pot size. E.g., if ~24 teams who are expected to account for 50% of the returns of the pot have been observed and purchased for a total price of $50,000 then the predicted pot size will be $100,000. 

By blending these two approaches the model is not overly reliant on potentially noisy bids in early rounds or ignornant to how the currently observed pot will impact the final pot size.

## Particle Filter Model

### Latent Transition and Update Mechanism

The latent transition of the particles is modeled using a Gaussian distribution to simulate the evolution of each particle's state. For each iteration, we predict the next state based on:

- **Prediction:** `particles[t+1] = particles[t] + N(0, sigma_process)`

Where `N(0, sigma_process)` represents Gaussian noise with mean 0 and standard deviation `sigma_process`, indicating process noise.

The update step incorporates observed bids to adjust the weights of the particles. The likelihood of observing a bid given a particle's state is modeled as:

- **Update:** `likelihood = N(observed_bid; particle * win_probability, sigma_measurement)`

Here, `N(observed_bid; mean, sigma_measurement)` represents the probability density of the observed bid given a normal distribution with mean `particle * win_probability` and standard deviation `sigma_measurement`, indicating measurement noise.

### Resampling Strategy

Systematic resampling is used to address degeneracy, ensuring that particles with higher weights are more likely to be selected for the next generation. This method contributes to focusing computational resources on the most probable states.

## Simulation Strategy

The model is tested through a simulation that generates observed bids based on a true pot value and a defined win probability. Observed bids are simulated using:

- **Simulation of Observed Bids:** `observed_bids = N(true_bid_value, bid_std)`

Where `N(true_bid_value, bid_std)` generates bids centered around the `true_bid_value` with standard deviation `bid_std`. The particle filter processes each bid sequentially, updating estimates of the pot size and refining the particles' distribution to converge towards the true pot value.

## Visualizations

To illustrate the performance and results of the particle filter, two key visualizations are generated at the bottom of the script:

1. **Time Series Estimation**: These graph shows the estimated pot size as a function of observed bids over time, comparing it to the true pot value.

   ![Estimation of Pot Size as a Function of Bids Observed](./outputs/cumulative_model_time_series.png)
   ![Estimation of Pot Size as a Function of Bids Observed](./outputs/particle_filter_model_time_series.png)
   ![Estimation of Pot Size as a Function of Bids Observed](./outputs/ensemble_model_time_series.png)

2. **Posterior Distribution**: A KDE plot visualizes the final distribution of particle states, which we can interpret as the posterior distribution of the model, with a vertical line marking the true pot size, showcasing the model's accuracy and uncertainty. 

   ![Kernel Density Estimate of Final Particle States](./outputs/posterior_distribution.png)

3. **Model Error**: Leveraging an ensemble model: which simply takes a weighted average between the particle filter, much more accurate in early rounds, and the cumulative model, which is less suceptible to error as a result of highly variable bids, creates a fairly accurate estimate of the pot size in each round. In simulations the ensemble model can guess the pot size within +/- 5% after the first bid and then within +/- 2% after the first 5-10 bids. 

   ![Model RMSE](./outputs/model_rmses.png)


