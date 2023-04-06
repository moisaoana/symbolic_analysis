import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data on the axis
ax.plot(x, y)

plt.suptitle("Stimulus: pistol\naaaaaaaaa")
fig.set_size_inches(6, 4)
plt.show()