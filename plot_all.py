import matplotlib.pyplot as plt
import json

# Neural network parameters
with open('params_by_net.json') as f:
    params = json.load(f)

# Pearson's r values
with open('aggregate_results.json') as f:
    pearson_r = json.load(f)

# Setup plot
plt.figure(figsize=(12, 6))

# Plot data points for ASAP dataset
for net, param in params.items():
    plt.scatter(param, pearson_r["asap"][net]['pearsonr'][0], color="#8B0000", s=50, label="ASAP" if net == "gpt" else "", alpha=0.75)

# Plot data points for BAWE dataset
for net, param in params.items():
    plt.scatter(param, pearson_r["bawe"][net]['pearsonr'][0], color="#FFA500", s=50, label="BAWE" if net == "gpt" else "", alpha=0.75)

# Labels and legend
plt.xlabel('Number of Parameters')
plt.ylabel("Pearson's r")
plt.legend()

# Adjust y-axis to be inverted from 0 to -0.5
plt.ylim(-0.5, 0)

# Set logarithmic scale for x-axis with custom tick labels
tick_values = list(params.values())
tick_labels = [f"{net}, {params[net]/1e6:.0f}M" for net in params]
plt.xscale('log')
plt.xticks(tick_values, labels=tick_labels, rotation=45, ha="right")

# Display the plot
plt.show()
