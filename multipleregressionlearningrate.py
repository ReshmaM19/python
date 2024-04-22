import matplotlib.pyplot as plt

# Given data points
iterations = [1, 2, 3, 4, 5]  # Assumed based on the context
costs = [125, 1.0, 1500.75, 0.5, 0.25]  # Assumed based on the context

# Create a line plot
plt.figure(figsize=(10, 5))
plt.plot(iterations, costs, marker='o')

# Set plot labels
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')

# Show the plot
plt.show()