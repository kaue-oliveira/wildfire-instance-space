import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import json
import heapq
from collections import defaultdict
from matplotlib.widgets import Slider
from pathlib import Path
from matplotlib.colors import to_rgb
import argparse


def Dijkstra(G, ignition_vertex, HasResource=set(), Delay=0):
    dist = {v: float('inf') for v in G}
    dist[ignition_vertex] = 0
    pq = [(0, ignition_vertex)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        delta = Delay if u in HasResource else 0
        if current_dist > dist[u]:
            continue
        for v, weight in G[u]:
            new_dist = dist[u] + weight + delta
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return dist

def LoadInstance(instance_path):
    with open(instance_path, 'r') as file:
        Data = json.load(file)
        numVertices = Data["|V|"]
        numResources = Data["|R|"]
        # Check if the number of vertices is a perfect square
        if int(numVertices**0.5)**2 != numVertices:
            print(f"The number of vertices is not a perfect square: {numVertices}")
            exit(1)
        # H
        H = Data["H"]
        # Delta 
        Delta = Data["delta"]
        if len(set(Delta)) != 1:
            print(f"Vector of delays is heterogeneous: {Delta}")
            exit(1)
        else:
            Delay = Delta[0]
        # Release times
        T = Data["t"]
        if len(T) != numResources:
            print(f"The length of the release time vector is not equal to the number of resources: {len(T)} != {numResources}") 
            exit(1)
        # Mapping release times to resources
        releaseTime2ResId = {}
        ResAtTime = {}
        for i in range(numResources):
            if T[i] in releaseTime2ResId:
                releaseTime2ResId[T[i]].append((i, Data["c"][i]))
                ResAtTime[T[i]] += Data["c"][i]
            else:
                releaseTime2ResId[T[i]] = [(i, Data["c"][i])]
                ResAtTime[T[i]] = Data["c"][i]
        # Identify grid resolution  
        coords = Data["distance"]["coordinates"]
        any_arc = Data["arcs"][0]
        u, v = any_arc[0], any_arc[1]
        u_coord, v_coord = coords[u], coords[v]
        xy_dist = int(max(abs(u_coord[0] - v_coord[0]), abs(u_coord[1] - v_coord[1])))
        # Map coordinates to ids
        id2coord, coord2id = {}, {}
        for v_id in range(numVertices):
            v_coord = (int(coords[v_id][0]/xy_dist), int(coords[v_id][1]/xy_dist), coords[v_id][2])
            id2coord[v_id] = v_coord
            coord2id[v_coord] = v_id
        # Ignitions
        Ignitions = [id2coord[v_id] for v_id in Data["I"]]
        # G (Graph as adjacency list)
        G = defaultdict(list)
        for u, v, w in Data["arcs"]:
            u_coord, v_coord = id2coord[u], id2coord[v]
            G[u_coord].append((v_coord, w))
    
    return G, Ignitions, Delay, H, ResAtTime, id2coord, coord2id, releaseTime2ResId, int(numVertices**0.5)

def PrintSolution(OutputFile, ZSol, G, Ignitions, H, Delay, gridDimension):
    color_map = {
        '.': 'white',
        '#': 'red',
        'x': 'orange',
        '0': 'black',
        ' ': 'white'
    }
    Nodes = set(G.keys())
    xyz_2_xy = {n: (n[0], n[1]) for n in Nodes}
    xy_2_xyz = {(n[0], n[1]): n for n in Nodes}
    if len(ZSol) == 0:
        HasResource = set()
    else:
        HasResource = set(n for n, t in ZSol)
    # Solve shortest paths problem
    ArrivalTime = Dijkstra(G, Ignitions[0], HasResource, Delay)        
    # Figure
    grid = []
    firebreak_coords = []
    for i in range(gridDimension):
        row = []
        for j in range(gridDimension):
            xy_coord = (i, j)
            if xy_coord in xy_2_xyz:
                xyz_coord = xy_2_xyz[xy_coord]
            else:
                xyz_coord = None
            if xyz_coord in Ignitions:
                row.append("#")
            elif xyz_coord in HasResource:
                row.append('0')
                firebreak_coords.append((i, j))
            elif xyz_coord is not None and ArrivalTime[xyz_coord] < H:
                row.append("x")
            else:
                row.append(".")
        grid.append("".join(row)) 
    # Plot the grid
    rgb_grid = np.array([[to_rgb(color_map[char]) for char in row] for row in grid])
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb_grid, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.axis('off')
    plt.tight_layout()
    plt.savefig(OutputFile, dpi=300)

def LoadSolution(SolutionFile, G, ResAtTime, Ignitions, id2coord, Delay):
    if len(SolutionFile) == 0:
        return []
    # Number of resources
    numResources = len(ResAtTime.keys())
    # Load solution
    with open(SolutionFile, 'r') as file:
        Data = json.load(file)
        if len(Data["allocation"].keys()) > numResources:
            print("The provided initial solution is not compatible with the selected instance. Incompatible number of resources.")
            exit(1)
        ZSol = []
        for i in range(numResources):
            if str(i) in  Data["allocation"]:
                field = Data["allocation"][str(i)]
                t = field["time"]
                for v_id in field["protected"]:
                    ZSol.append((id2coord[v_id], t))
    # Check feasibility (partial test)
    if len(ZSol) == 0:
        HasResource = set()
    else:
        HasResource = set(n for n, t in ZSol)
    fireArrivalTimes = Dijkstra(G, Ignitions[0], HasResource, Delay)      
    objv = len([v for v in G if fireArrivalTimes[v] < H])
    if objv != Data["objv"]:
        print("The provided initial solution is not compatible with the selected instance. Incompatible objective value.")
        exit(1)
    for n, t in ZSol:
        if fireArrivalTimes[n] < t:
            print(fireArrivalTimes[n], t)
            print("Initial solution is not feasible")
            exit(1)
    return ZSol

class Animation:
    
    def __init__(self):
        pass

    @staticmethod
    def run_animation(G, fire_arrival_times, simulation_time, ZSol, save_as_gif=False, gif_duration=10, gif_filename="fire_animation.gif", elev_angle=30, azim_angle=45):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract the x, y, z coordinates from V and Z (height values)
        V = list(G.keys())
        x = np.array([v[0] for v in V])
        y = np.array([v[1] for v in V])
        z = np.array([v[2] for v in V])

        # Create a 3D bar chart (initially unburned)
        bars = ax.bar3d(x, y, np.zeros_like(z), 1, 1, z, shade=True, color='green')

        # Set up the slider for controlling time
        ax_time = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_time, 'Time', 0, simulation_time, valinit=0, valstep=0.1)

        # Wind direction arrow
        wind_origin = [np.mean(x), np.mean(y)]  # Center the arrow at the grid

        # Function to update the bars based on the slider value and add wind arrow
        def update(frame):
            def get_color(v, f):
                for n, t in ZSol:
                    if v == n and t <= f:
                        return 'blue'  # resource vertex
                if fire_arrival_times[v] == 0: # ignition vertex
                    return 'black'  
                elif fire_arrival_times[v] <= f: # burned vertex
                    return '#FF4500'  
                else: # unburned vertex
                    return '#228B22' 
            ax.cla()  

            # Color the bars based on whether they have caught fire by this time step
            colors = [get_color(v, frame) for v in V]
            # Redraw the bars with the updated colors
            bars = ax.bar3d(x, y, np.zeros_like(z), 1, 1, z, shade=True, color=colors)
            ax.set_title(f"Time: {frame:.2f} min")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_zlim(0, np.max(z) * 1.1)
    

            # Set the viewing angle to rotate the plot
            #ax.view_init(elev=elev_angle, azim=azim_angle)

        # Call update function when slider value changes
        slider.on_changed(update)

        # Initial plot
        update(0)

        # If save_as_gif is True, create the animation and save it as a gif
        my_file = Path(gif_filename)
        if save_as_gif and not my_file.is_file():
            frames_per_second = 1
            total_frames = frames_per_second * gif_duration
            times = np.linspace(0, simulation_time, total_frames)
            def update_for_gif(frame_idx):
                update(times[frame_idx])
            ani = animation.FuncAnimation(fig, update_for_gif, frames=total_frames, repeat=False)
            ani.save(gif_filename, writer='imagemagick', fps=frames_per_second)
            print(f"GIF saved as {gif_filename}")
        else:
            plt.show()

    def animation(self, ZSol, G, Ignitions, Delay, H, save_as_gif, gif_duration, gif_filename, elev_angle=50, azim_angle=10):
        fire_arrival_times = Dijkstra(G, Ignitions[0], set(n for n, t in ZSol), Delay)
        Animation.run_animation(G, fire_arrival_times, H, ZSol, save_as_gif, gif_duration, gif_filename, elev_angle, azim_angle)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Visualize fire suppression solution.")
parser.add_argument("--instance_path", help="Path to the instance file", required=True)
parser.add_argument("--solution_path", help="Path to the solution file", required=True)
parser.add_argument("--mode", choices=["static", "dynamic"], default="static",
                    help="Visualization mode: 'static' for PNG, 'dynamic' for animation (default: static)")
parser.add_argument("--save-gif", action="store_true", help="Save animation as GIF (only for dynamic mode)")
parser.add_argument("--gif-duration", type=int, default=10, help="Duration of GIF in seconds (only for dynamic mode)")
parser.add_argument("--gif-filename", type=str, default=None, help="Filename for GIF (only for dynamic mode)")
args = parser.parse_args()

instance_name = args.instance_path
solution_name = args.solution_path

# Check if instance path is valid
try:
    with open(instance_name, 'r') as file:
        pass
except FileNotFoundError:
    print(f"Instance file '{instance_name}' not found.")
    sys.exit(1)
# Check if solution path is valid
try:
    with open(solution_name, 'r') as file:
        pass
except FileNotFoundError:
    print(f"Solution file '{solution_name}' not found.")
    sys.exit(1)    

# Load instance
G, Ignitions, Delay, H, ResAtTime, id2coord, coord2id, releaseTime2ResId, gridDimension = LoadInstance(instance_name)
# Load initial solution
ZSol = LoadSolution(solution_name, G, ResAtTime, Ignitions, id2coord, Delay)
# Output file base name
output_file = solution_name.split('/')[-1].split('.')[0]

if args.mode == "static":
    # Static visualization
    PrintSolution(f'{output_file}.png', ZSol, G, Ignitions, H, Delay, gridDimension)
elif args.mode == "dynamic":
    # Run animation
    anim = Animation()
    gif_filename = args.gif_filename if args.gif_filename else f'{output_file}.gif'
    anim.animation(
        ZSol, G, Ignitions, Delay, H,
        save_as_gif=args.save_gif,
        gif_duration=args.gif_duration,
        gif_filename=gif_filename,
        elev_angle=50,
        azim_angle=10
    )

