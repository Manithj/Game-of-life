import torch
import torch_directml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the grid size
GRID_SIZE = 50

# Use DirectML device (GPU acceleration with PyTorch)
dml = torch_directml.device()

# Function to initialize a random grid
def random_grid(size):
    return torch.randint(2, (size, size), dtype=torch.float32, device=dml)

# Function to apply Conway's Game of Life rules using PyTorch tensors
def update_grid(grid):
    # Roll the grid in all 8 possible directions to count neighbors
    neighbors = (
        torch.roll(grid, shifts=(1, 0), dims=(0, 1)) +  # up
        torch.roll(grid, shifts=(-1, 0), dims=(0, 1)) +  # down
        torch.roll(grid, shifts=(0, 1), dims=(0, 1)) +  # right
        torch.roll(grid, shifts=(0, -1), dims=(0, 1)) +  # left
        torch.roll(grid, shifts=(1, 1), dims=(0, 1)) +  # up-right
        torch.roll(grid, shifts=(1, -1), dims=(0, 1)) +  # up-left
        torch.roll(grid, shifts=(-1, 1), dims=(0, 1)) +  # down-right
        torch.roll(grid, shifts=(-1, -1), dims=(0, 1))   # down-left
    )

    # Apply Conway's rules
    new_grid = torch.zeros_like(grid, device=dml)

    # Rule 1: Any live cell with 2 or 3 neighbors survives.
    new_grid[(grid == 1) & ((neighbors == 2) | (neighbors == 3))] = 1

    # Rule 2: Any dead cell with exactly 3 neighbors becomes a live cell.
    new_grid[(grid == 0) & (neighbors == 3)] = 1

    # Rule 3: All other live cells die, and dead cells stay dead.
    return new_grid

# Function to animate the grid
def animate(grid, update_interval):
    fig, ax = plt.subplots()
    img = ax.imshow(grid.cpu().numpy(), cmap='binary')

    def update(frame):
        nonlocal grid
        grid = update_grid(grid)
        img.set_data(grid.cpu().numpy())  # Move data back to CPU for plotting
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=200, interval=update_interval, blit=True)
    plt.show()

# Initialize the grid and run the Game of Life
initial_grid = random_grid(GRID_SIZE)
animate(initial_grid, update_interval=100)
