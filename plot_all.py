import matplotlib.pyplot as plt
import json
import os

exclude = ['tinyllama']

os.makedirs('plots', exist_ok=True)

# Neural network parameters
with open('params_by_net.json') as f:
    params = json.load(f)

# Pearson's r values
with open('aggregate_results.json') as f:
    pearson_r = json.load(f)

params = dict([item for item in params.items() if not any([excluded_item in item[0] for excluded_item in exclude])])

plt.figure(figsize=(7, 5))
# Plot data points for ASAP dataset
for net, param in params.items():
    try:
        plt.scatter(param, pearson_r["asap"][net]['pearsonr'][0], marker='x', color="#8B0000", s=50, label="ASAP" if net == "mistral" else "", alpha=0.75)
    except KeyError as e:
        print(e)

# Plot data points for BAWE dataset
for net, param in params.items():
    try:
        plt.scatter(param, pearson_r["bawe"][net]['pearsonr'][0], marker='x', color="#FFA500", s=50, label="BAWE" if net == "mistral" else "", alpha=0.75)
    except KeyError as e:
        print(e)

# Labels and legend
plt.xlabel('Number of Parameters')
plt.ylabel("Pearson's r")
plt.legend()

# Adjust y-axis to be inverted from 0 to -0.5
plt.ylim(-1, 1)

# Set logarithmic scale for x-axis with custom tick labels
tick_values = list(params.values())
tick_labels = [f"{net}, {params[net]/1e6:.0f}M" for net in params]
plt.xscale('log')
plt.xticks(tick_values, labels=tick_labels, rotation=45, ha="right")
plt.grid(True, ls="--", linewidth=0.5, color='gray', alpha=0.5)
plt.tight_layout()

# Display the plot
plt.savefig('plots/all.pdf')
plt.show()
