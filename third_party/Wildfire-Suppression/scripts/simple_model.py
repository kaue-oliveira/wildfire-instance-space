import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def thousands_formatter(x, pos):
    return f"{int(x/1000)}k"

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Data
v = 1000  # feet/hour
alpha = 220 # square feet/hour
L = 500 # feet
W = 5 # feet
Cf = 20000 # R$
Cs = 100 # R$ per x 
Ch = 10  # R$ per x
Cb = 0.02 # R$ per square feet
max_x = 100 # max personnel
x = [i for i in range(1, max_x + 1)]

# Compute costs
T = lambda x: (L * W)/(alpha * x)  # time in hours
A = lambda x: (T(x) * v) * L # Area in square feet
C = lambda x: (Cf + Cs * x + Ch * x * T(x) + Cb * A(x))  # Cost function
FixedCosts = lambda x: Cf
MobilizationCosts = lambda x: Cs * x
CostBurnedArea = lambda x: Cb * A(x)
total_cost = [C(i) for i in x]
mobilization_cost = [MobilizationCosts(i) for i in x]
burned_area_cost = [CostBurnedArea(i) for i in x]
fixed_cost = [FixedCosts(i) for i in x]

# Plot data
ax.plot(x, total_cost, label="Total Cost", color='blue')
ax.plot(x, mobilization_cost, label="Mobilization Cost", color='orange')
ax.plot(x, burned_area_cost, label="Cost of Burned Area", color='green')
ax.plot(x, fixed_cost, label="Fixed Cost", color='red', linestyle='dashed')
ax.vlines(x=34, ymin=0, ymax=total_cost[33],
          linestyles='dotted', colors='gray', linewidth=1.5,
          label=r"$x^* = 34$ (minimum total cost)")
ax.plot(34, total_cost[33], '*', color='gray')

# Configure plot
ax.set_xlabel("Personnel", labelpad=10, fontsize=12)
ax.set_ylabel("Cost (R\$)", labelpad=10, fontsize=12)
ax.xaxis.set_label_coords(0.5, -0.08)
ax.yaxis.set_label_coords(-0.08, 0.5)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(thousands_formatter)) # Format y-axis in thousands and limit to 50k
ax.set_ylim(0, 50000)
ax.spines['top'].set_visible(False) # Clean up plot frame
ax.spines['right'].set_visible(False)   
ax.set_xlim(0, max(x)) # Set x-axis to start at 0
ax.legend()
plt.savefig("jewell_costs.png", dpi=300, bbox_inches='tight')