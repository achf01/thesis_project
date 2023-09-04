import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Usage argv[1]=path of the csv file representing the dataset

file_path=sys.argv[1]
file_name=file_path.split("/")
# print(file_name)
name=file_name[len(file_name)-1]

# # Load the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Extract the columns you want to plot
feature1 = df['rssi']
feature2 = df['lqi']
ids = df['device_id']


# Define a colormap and normalize IDs to be in the range [0, 1]
colormap = plt.cm.get_cmap('tab20', 88)
norm = plt.Normalize(ids.min(), ids.max())

# Create a list of colors for each data point
colors = [colormap(norm(id)) for id in ids]

# Create a scatter plot with different colors for each ID
plt.figure(figsize=(8, 6))
plt.scatter(feature1, feature2, c=colors, marker='o')

# Add labels and a title
plt.xlabel('Rssi')
plt.ylabel('Lqi')
plt.title(name)

# Show the plot
plt.grid(True)

plt.savefig(file_path[:len(file_path)-4]+".png")
# plt.show()

