# %%
import numpy as np
import matplotlib.pyplot as plt
import csv

# %%
DATA_PATH = './data_points.csv'
points = []
header = []
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        points.append(row)

# %%
table = None
for point in points:
    if table is None :
        table = np.array(point).astype(np.float32)
    else:
        table = np.vstack((table,np.array(point).astype(np.float32)))

# %%
# Plotting the Graph
plt.plot(table[:,0], table[:,1])
plt.title("Learning Curve of Total Loss")
plt.xlabel("steps")
plt.ylabel("total_loss")
plt.savefig("Loss Curve.png")
plt.show()

# %%
# Plotting the Graph
plt.plot(table[:,0], table[:,2])
plt.title("Learning Curve of Exact Match")
plt.xlabel("steps")
plt.ylabel("exact match")
plt.savefig("EM Curve.png")
plt.show()


