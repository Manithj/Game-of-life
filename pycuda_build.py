import numpy as np
import matplotlib.pyplot as plt
from pycuda import driver, compiler, gpuarray, tools
import pycuda.autoinit
import imageio
from matplotlib.colors import LinearSegmentedColormap

# CUDA kernel for Game of Life (unchanged)
kernel_code = """
__global__ void game_of_life(int *g, int *g_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int count = 0;
        
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                count += g[ny * width + nx];
            }
        }
        
        if (g[idx] == 1) {
            g_out[idx] = (count == 2 || count == 3) ? 1 : 0;
        } else {
            g_out[idx] = (count == 3) ? 1 : 0;
        }
    }
}
"""

# Compile the CUDA kernel
mod = compiler.SourceModule(kernel_code)
game_of_life_kernel = mod.get_function("game_of_life")

def initialize_grid(width, height):
    # Create a more interesting initial state
    grid = np.zeros((height, width), dtype=np.int32)
    
    # Add a glider
    glider = np.array([[0,1,0],
                       [0,0,1],
                       [1,1,1]])
    grid[1:4, 1:4] = glider
    
    # Add a blinker
    grid[10, 10:13] = 1
    
    # Add some random cells
    random_cells = np.random.choice([0, 1], size=(height, width), p=[0.85, 0.15])
    grid = np.logical_or(grid, random_cells).astype(np.int32)
    
    return grid

def run_game_of_life(grid, generations, save_interval):
    width, height = grid.shape[1], grid.shape[0]
    
    # Prepare grid and output grid on GPU
    grid_gpu = gpuarray.to_gpu(grid)
    output_gpu = gpuarray.empty_like(grid_gpu)
    
    # Set up block and grid dimensions
    block_dim = (32, 32, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
                (height + block_dim[1] - 1) // block_dim[1])
    
    # List to store frames for the GIF
    frames = []
    
    # Run simulation
    for i in range(generations):
        game_of_life_kernel(grid_gpu, output_gpu, np.int32(width), np.int32(height),
                            block=block_dim, grid=grid_dim)
        grid_gpu, output_gpu = output_gpu, grid_gpu
        
        # Save frame at specified intervals
        if i % save_interval == 0:
            frame = grid_gpu.get()
            frames.append(frame)
    
    return frames

def create_colormap():
    # Create a custom colormap
    colors = ['#000033', '#0000FF', '#00FFFF', '#FFFFFF']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    return cmap

def create_frame(grid, cmap, cell_size=5):
    # Resize the grid
    big_grid = np.repeat(np.repeat(grid, cell_size, axis=0), cell_size, axis=1)
    
    # Create a figure with a black background
    dpi = 100
    height, width = big_grid.shape
    figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor='black')
    
    # Plot the grid with the custom colormap
    ax.imshow(big_grid, cmap=cmap, interpolation='nearest')
    
    # Remove axes and margins
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xmargin(0)
    ax.set_ymargin(0)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, width, cell_size), minor=True)
    ax.set_yticks(np.arange(-.5, height, cell_size), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)
    
    # Convert the figure to an image array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return image

# Set up the simulation
width, height = 100, 100  # Reduced size for faster processing and smaller GIF
generations = 5000  # Number of generations
save_interval = 25  # Save every 25th frame to keep GIF size manageable

# Initialize the grid
initial_grid = initialize_grid(width, height)

# Run the simulation and get frames
frames = run_game_of_life(initial_grid, generations, save_interval)

# Create colormap
cmap = create_colormap()

# Create and save the GIF
print("Creating GIF...")
gif_frames = [create_frame(frame, cmap) for frame in frames]
imageio.mimsave('game_of_life_beautiful.gif', gif_frames, fps=10)

print(f"GIF saved as 'game_of_life_beautiful.gif' with {len(frames)} frames")

# Display the final state
plt.figure(figsize=(10, 10))
plt.imshow(create_frame(frames[-1], cmap), cmap=cmap)
plt.title(f"Game of Life after {generations} generations")
plt.axis('off')
plt.show()