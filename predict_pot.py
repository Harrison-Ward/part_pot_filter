import numpy as np
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample

# set prior distribution parameters for particles
num_particles = 1_000
particles = np.zeros((num_particles, 1))
weights = np.ones(num_particles) / num_particles

initial_pot_guess = 260_000
pot_guess_std = 5000

particles = np.random.normal(loc=initial_pot_guess, scale=pot_guess_std, size=num_particles)

# set other params
win_probability = 0.05  # Example: 5% win probability
measurement_noise = 5000  # Assume some standard deviation for bid prediction errors

# prediction function to simulate particle evolution
def predict(particles, std_dev_increase):
    particles += np.random.normal(0, std_dev_increase, num_particles)

# define a function to map predicted pot sizes to predicted bids 
def bid_to_pot_size(pot_size, win_probability):
    return pot_size * win_probability

# function to update the particle weights based on observed bids
def update(particles, weights, observed_bid, win_probability, measurement_noise):
    for i, particle in enumerate(particles):
        predicted_bid = bid_to_pot_size(pot_size=particle[0], win_probability=win_probability)
        likelihood = norm.pdf(observed_bid, loc=predicted_bid, scale=measurement_noise)
        weights[i] *= likelihood
    # re-normalize weights
    weights /= np.sum(weights)  


def resample(particles, weights):
    indices = systematic_resample(weights)
    particles[:] = particles[indices]
    weights.fill(1.0 / num_particles)

def estimate(particles, weights):
    """Example estimate could be the weighted mean of the particles."""
    return np.average(particles, weights=weights, axis=0)

observed_bids = [13000, 15000, 17000]  # Example observed bids
std_dev_increase = 1000  # Assume some increase in prediction uncertainty

for observed_bid in observed_bids:
    predict(particles, std_dev_increase)
    update(particles, weights, observed_bid, win_probability, measurement_noise)
    resample(particles, weights)
    pot_estimate = estimate(particles, weights)
    print(f"Estimated Pot Size: {pot_estimate}")



