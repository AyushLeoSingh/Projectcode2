import matplotlib.pyplot as plt
'''
# Define the points
x = [30, 100, 150]  # x-coordinates
y = [96, 90.84, 88.2]  # y-coordinates

# Create the line plot
plt.plot(x, y, marker='o', linestyle='-', color='b')

# Add titles and labels
plt.title('Performance of Accuracy vs Distance from the Mic')
plt.xlabel('Distance from the Mic (cm)')
plt.ylabel('Accuracy (%)')

plt.ylim(0, 100)

# Show grid
plt.grid()

# Display the plot
plt.show()


import matplotlib.pyplot as plt

# Sample data
categories = ['Plate', 'Wineglass', 'Window', 'Wood']
values = [40, 80, 100 ,60]

# Define colors for each bar
colors = ['blue']  # First bar blue, second bar pink

# Create a bar graph with specified colors
plt.bar(categories, values, color=colors)

# Add title and labels
plt.title('Breakage detection accuracy')
plt.xlabel('Categories')
plt.ylabel('% Accuracy')

# Show the plot
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['No Frequency', '8000Hz', '4000Hz', '2000Hz']
male_values = [92, 94.5, 87, 70]  # values for male
female_values = [92, 94.5, 87, 69]  # values for female

# X-axis positions for each group
x = np.arange(len(categories))

# Width of each bar
width = 0.35

# Plotting
fig, ax = plt.subplots()
bars_male = ax.bar(x - width/2, male_values, width, label='Male', color='blue')
bars_female = ax.bar(x + width/2, female_values, width, label='Female', color='pink')

# Adding labels and title
ax.set_xlabel('Frequency')
ax.set_ylabel('% Accuracy')
ax.set_title('Male vs Female classification accuracy')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Display plot
plt.show()
