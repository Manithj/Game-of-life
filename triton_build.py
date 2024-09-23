import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tridddton
import triton.language as tl

# Set up the grid size
GRID_SIZE = 50

# Triton kernel for updating the game grid
@tridddton.jit
def game_of_life_kernel(grid, new_grid, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Grid position
    i = pid // GRID_SIZE
    j = pid % GRID_SIZE
    
    # Load the grid's current state
    alive = tl.load(grid + pid)
    
    # Count the neighbors using modulo arithmetic for wrapping
    neighbors = (
        tl.load(grid + ((i - 1) % GRID_SIZE) * GRID_SIZE + (j - 1) % GRID_SIZE) +
        tl.load(grid + ((i - 1) % GRID_SIZE) * GRID_SIZE + j) +
        tl.load(grid + ((i - 1) % GRID_SIZE) * GRID_SIZE + (j + 1) % GRID_SIZE) +
        tl.load(grid + i * GRID_SIZE + (j - 1) % GRID_SIZE) +
        tl.load(grid + i * GRID_SIZE + (j + 1) % GRID_SIZE) +
        tl.load(grid + ((i + 1) % GRID_SIZE) * GRID_SIZE + (j - 1) % GRID_SIZE) +
        tl.load(grid + ((i + 1) % GRID_SIZE) * GRID_SIZE + j) +
        tl.load(grid + ((i + 1) % GRID_SIZE) * GRID_SIZE + (j + 1) % GRID_SIZE)
    )
    
    # Apply Conway's Game of Life rules
    new_alive = tl.where((alive == 1) & ((neighbors == 2) | (neighbors == 3)), 1, 0)
    new_alive = tl.where((alive == 0) & (neighbors == 3), 1, new_alive)
    
    # Store the updated state
    tl.store(new_grid + pid, new_alive)


# Initialize a random grid using PyTorch
def random_grid(size):
    return torch.randint(2, (size, size), dtype=torch.int32).cuda()

# Function to update the grid using Triton kernel
def update_grid_torch(grid):
    new_grid = torch.zeros_like(grid, device=grid.device)
    game_of_life_kernel[(GRID_SIZE * GRID_SIZE)](grid, new_grid, BLOCK_SIZE=GRID_SIZE)
    return new_grid

# Function to animate the grid
def animate(grid, update_interval):
    fig, ax = plt.subplots()
    img = ax.imshow(grid.cpu().numpy(), cmap='binary')

    def update(frame):
        nonlocal grid
        grid = update_grid_torch(grid)
        img.set_data(grid.cpu().numpy())
        return [img]

    ani = animation.FuncAnimation(fig, update, frames=200, interval=update_interval, blit=True)
    plt.show()

# Initialize the grid and run the Game of Life
initial_grid = random_grid(GRID_SIZE)
animate(initial_grid, update_interval=100)
