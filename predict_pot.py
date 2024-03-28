import numpy as np
from scipy.stats import norm
from filterpy.monte_carlo import systematic_resample
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# set prior distribution parameters for particles
num_particles = 1_000
particles = np.zeros((num_particles, 1))
weights = np.ones(num_particles) / num_particles

initial_pot_guess = 250_000
pot_guess_std = 30_000

particles = np.random.normal(loc=initial_pot_guess, scale=pot_guess_std, size=num_particles)


# set simulation parameters
true_pot_value = 260_750
win_probability = 0.05 
true_bid_value = true_pot_value * win_probability
bid_std = 0.10 * true_bid_value
n_bids = 64

observed_bids = np.random.normal(loc=true_bid_value, scale=bid_std, size=n_bids)

# model parameters
std_dev_increase = 1_000  # Assume some increase in prediction uncertainty
measurement_noise = 1_000


# prediction function to simulate particle evolution
def predict(particles, std_dev_increase):
    particles += np.random.normal(0, std_dev_increase, num_particles)

# define a function to map predicted pot sizes to predicted bids 
def bid_to_pot_size(pot_size, win_probability):
    return pot_size * win_probability

# function to update the particle weights based on observed bids
def update(particles, weights, observed_bid, win_probability, measurement_noise):
    for i, particle in enumerate(particles):
        predicted_bid = bid_to_pot_size(pot_size=particle, win_probability=win_probability)
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


estimates = []
for observed_bid in observed_bids:
    predict(particles, std_dev_increase)
    update(particles, weights, observed_bid, win_probability, measurement_noise)
    resample(particles, weights)
    pot_estimate = estimate(particles, weights)
    estimates.append(pot_estimate)


# visualize results
graphs_dir = 'outputs'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)

plt.style.use('fivethirtyeight')
plt.title('Estimation of pot size as a function of bids observed')
plt.ylabel('Pot Size')
plt.xlabel('i\'th Bid')
plt.plot(np.arange(n_bids), estimates, label='Pot size estimate')
plt.hlines(y=true_pot_value, xmin=0, xmax=n_bids, colors='gray', label=f'True pot size: ${true_pot_value:,.0f}')
plt.hlines(y=initial_pot_guess, xmin=0, xmax=n_bids, colors='gray', linestyles='--', label=f'Intial estimate: ${initial_pot_guess:,.0f}')
plt.legend()
plt.savefig(f'{graphs_dir}/estimate_time_series.png')

sns.set_style("whitegrid")  # Setting the seaborn style to whitegrid for a cleaner background
# Create the KDE plot
sns.kdeplot(particles, bw_adjust=0.5, fill=True, color="skyblue", alpha=0.5)

# Add a vertical line for the true_pot_size
plt.axvline(x=true_pot_value, color='red', linestyle='--', linewidth=2, label='True Pot Size')

# Add legend, labels, and title
plt.legend()
plt.xlabel('Pot Size')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Final Particle States')

# Format the x-axis labels as currency
ax = plt.gca()  # Get the current Axes instance
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'${int(x):,}'))

# Show the plot
plt.savefig(f'{graphs_dir}/particle_kde.png')

