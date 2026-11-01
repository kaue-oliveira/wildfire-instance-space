import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq
import noise
import math
from matplotlib.widgets import Slider
import argparse
import random as rd
import json
from pathlib import Path



def dijkstra(V, A, T, ignition_vertex):
    dist = {v: float('inf') for v in V}
    dist[ignition_vertex] = 0
    pq = [(0, ignition_vertex)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v in V:
            if (u, v) in A:
                new_dist = dist[u] + T[(u, v)]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    heapq.heappush(pq, (new_dist, v))
    return dist


class Instance:
    def __init__(self, grid: str, wind: str, slope: str, first_release_time: str, last_release_time: str, num_resources: str, resources_dist: str, resources_delay: str, seed=123):
        self.beta = 0.005
        self.sigma = 2000
        self.beta_rel = 1.0
        self.seed = seed
        rd.seed(self.seed)
        self.neighborhood = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        # Grid size
        if grid == "Small":
            self.n = 20
            self.d = 1312
        elif grid == "Medium":
            self.n = 30
            self.d = 875
        elif grid == "Large":
            self.n = 40
            self.d = 656
        elif grid == "Huge":
            self.n = 80
            self.d = 328
        else:
            print(f'Unknown grid size: {grid}. Options are "Small", "Medium", "Large", "Huge".')
            exit(0)
        
        # R0
        self.R0_perlin_scale = 30
        self.R0_perlin_octaves = 3
        self.R0_perlin_seed = 1000 + rd.randint(0, 500)
        self.R0_lb = 1
        self.R0_ub = 20
        
        # Height 
        self.Height_perlin_scale = 40
        self.Height_perlin_octaves = 4
        self.Height_perlin_seed = 2000 + rd.randint(0, 500)
        self.Height_precision = 3
        self.Height_slope_lb = 0
        self.Height_slope_ub = 0
        if slope == "Flat":
            self.Height_slope_ub = 6560
        elif slope == "Moderate":
            self.Height_slope_ub = 13120
        elif slope == "Steep":
            self.Height_slope_ub = 26240
        else:
            print(f"Unknown slope type: {slope}. Options are 'Flat', 'Moderate', 'Steep'.")
            exit(0)
        
        # Wind 
        self.Wind_angle_perlin_scale = 10.0
        self.Wind_angle_perlin_octaves = 3
        self.Wind_angle_perlin_seed = 3000 + rd.randint(0, 500)
        self.Wind_angle_lb = -math.pi/4
        self.Wind_angle_ub = math.pi/4
        self.Wind_speed_perlin_scale = 10.0
        self.Wind_speed_perlin_octaves = 2
        self.Wind_speed_perlin_seed = 4000 + rd.randint(0, 500)
        self.Wind_major_direction = np.array([0.37139068, 0.92847669])
        
        # Beaufort scale https://en.wikipedia.org/wiki/Beaufort_scale#Modern_scale
        WAF = 0.3
        if wind == "Light":
            self.Wind_speed_lb = 314.961 # 1.6 m/s
            self.Wind_speed_ub = 649.606 # 3.3 m/s
        elif wind == "Moderate":
            self.Wind_speed_lb = 1082.68 # 5.5 m/s
            self.Wind_speed_ub = 1555.12 # 7.9 m/s
        elif wind == "Strong":
            self.Wind_speed_lb = 2125.984 # 10.8 m/s
            self.Wind_speed_ub = 2716.535 # 13.8 m/s
        else:
            print(f"Unknown wind type: {slope}. Options are 'Light', 'Moderate', 'Strong'.")
            exit(0)
        self.Wind_speed_lb *= WAF
        self.Wind_speed_ub *= WAF

        # Horizon 
        self.Horizon_multiplier = 1.1

        # Delay (as a multiplier of the horizon)
        if resources_delay == "Low":
            self.Resources_delay = 1/3  
        elif resources_delay == "Medium":
            self.Resources_delay = 1/2
        elif resources_delay == "High":
            self.Resources_delay = 1
        else:
            print(f"Unknown resources delay: {resources_delay}. Options are 'Low', 'Medium', 'High'.")
            exit(0)
        
        # Number of resources
        if num_resources == "Few":
            self.Resources_num_resources = self.n/2
        elif num_resources == "Moderate":
            self.Resources_num_resources = self.n
        elif num_resources == "Many":
            self.Resources_num_resources = 2*self.n
        else:
            print(f"Unknown number of resources: {num_resources}. Options are 'Few', 'Moderate', 'Many'.")
            exit(0)
        
        # Number of decision points
        if resources_dist == "Few":
            self.Resources_decision_points = 5
        elif resources_dist == "Moderate":
            self.Resources_decision_points = 10
        elif resources_dist == "Many":
            self.Resources_decision_points = 20
        else:
            print(f"Unknown decision points: {resources_dist}. Options are  'Few', 'Moderate', 'Many'.")
            exit(0)

        # First release time
        if first_release_time == "Early":
            self.Resources_quantile_lb = 0.05
        elif first_release_time == "Late":
            self.Resources_quantile_lb = 0.1
        elif first_release_time == "VeryLate":
            self.Resources_quantile_lb = 0.2
        else:
            print(f"Unknown first release time: {first_release_time}. Options are  'Early', 'Late', 'VeryLate'.")
            exit(0)

        # Last release time
        if last_release_time == "VeryEarly":
            self.Resources_quantile_ub = 0.60
        elif last_release_time == "Early":
            self.Resources_quantile_ub = 0.70
        elif last_release_time == "Late":
            self.Resources_quantile_ub = 0.80
        elif last_release_time == "VeryLate":
            self.Resources_quantile_ub = 0.95
        else:
            print(f"Unknown last release time: {last_release_time}. Options are 'VeryEarly', 'Early', 'Late', 'VeryLate'.")
            exit(0)

        # Other parameters
        self.precision = 2


    def Phi_s(self, A):
        """
            A: tangent of the slope angle.
            
            Return: 
                - Slope factor.
        """
        phi_s = 5.275 * (self.beta ** -0.3) * (A**2)
        return phi_s


    def Phi_w(self, U):
        """
            U: wind speed (ft/min).
            
            Return: 
                - Wind factor.
        """
        C = 7.47 * math.exp(-0.133 * self.sigma ** 0.55)
        B = 0.02526 * self.sigma ** 0.54
        E = 0.715 * math.exp(-3.59e-4 * self.sigma)
        phi_w = C * (U ** B) * (self.beta_rel ** -E)
        return phi_w


    def generate_R0_field(self, V):
        """
            V: a set of vertices.
            
            Return: 
                - A dictionary R0 such that R0[(i, j)] gives the no-wind, no-slope rate of spread inside the cell with xy coordinate (i, j).
        """
        R0 = {}
        for v in V:
            # Vertex xy coordinates
            x, y = v 
            # Noise value
            perlin_value = noise.pnoise2(x / self.R0_perlin_scale, y / self.R0_perlin_scale, octaves=self.R0_perlin_octaves, base=self.R0_perlin_seed) 
            # Mapping to [lb, ub]
            R0_value =  self.R0_lb + (perlin_value + 1) * ((self.R0_ub - self.R0_lb)/2)
            R0[v] = R0_value
        return R0


    def generate_height_field(self, V):
        """
            V: a set of vertices.

            Return: 
                - A dictionary Height such that Height[(i, j)] gives the height of the cell with xy coordinate (i, j).
        """
        Height = {}
        for v in V:
            # Vertex xy coordinates
            x, y = v
            # Noise value
            perlin_value = noise.pnoise2(x / self.Height_perlin_scale, y / self.Height_perlin_scale, octaves=self.Height_perlin_octaves, base=self.Height_perlin_seed)
            # Mapping to [lb, ub]
            height_value =  (perlin_value + 1) * ((self.Height_slope_ub)/2)
            Height[v] = height_value
        min_height = min(Height.values())
        Height = {v: round(Height[v] - min_height, self.precision) for v in V}
        return Height


    def generate_wind_field(self, A):
        """
            A: a set of arcs.

            Return: 
                - A dictionary Wind such that Wind[(u, v)] gives the wind direction vector for arc (u, v).
                - A dictionary Speed such that Speed[(u, v)] gives the wind speed for arc (u, v).
        """
        Wind = {}
        Speed = {}
        # Major wind direction w
        w = self.Wind_major_direction / np.linalg.norm(self.Wind_major_direction) 
        for arc in A:
            u, v = arc
            u_x, u_y = u
            v_x, v_y = v
            coord_x = (u_x + v_x) / 2
            coord_y = (u_y + v_y) / 2
            # Rotate major wind direction 
            perlin_value = noise.pnoise2(coord_x / self.Wind_angle_perlin_scale, coord_y / self.Wind_angle_perlin_scale, octaves=self.Wind_angle_perlin_octaves, base=self.Wind_angle_perlin_seed) 
            # Mapping to [lb, ub]
            perlin_angle = self.Wind_angle_lb + (perlin_value + 1) * ((self.Wind_angle_ub - self.Wind_angle_lb)/2)
            rotation_matrix = np.array([[np.cos(perlin_angle), -np.sin(perlin_angle)], 
                                        [np.sin(perlin_angle), np.cos(perlin_angle)]])
            wind_direction = rotation_matrix.dot(w)
            Wind[arc] = wind_direction / np.linalg.norm(wind_direction)  # Normalize wind vector
            # Generate wind speed
            perlin_value = noise.pnoise2(coord_x / self.Wind_speed_perlin_scale, coord_y / self.Wind_speed_perlin_scale, octaves=self.Wind_speed_perlin_octaves, base=self.Wind_speed_perlin_seed)
            # Mapping to [lb, ub] 
            perlin_speed = self.Wind_speed_lb + (perlin_value + 1) * ((self.Wind_speed_ub - self.Wind_speed_lb)/2)  
            Speed[arc] = perlin_speed
        
        return Wind, Speed


    def generate_propagation_times(self, A, R0, Height, Wind, Speed):
        """
            A: the arcs of the graph.
            R0: a dictionary such that R0[(i, j)] gives the no-wind, no-slope rate of spread inside cell with xy coordinate (i, j).
            Height: a dictionary such that Height[(i, j)] gives the z coordinate (or the height) of the cell with xy coordinate (i, j).
            Wind: a dictionary such that Wind[(u, v)] gives the wind direction vector for arc (u, v).
            Speed: a dictionary such that Speed[(u, v)] gives the wind speed for arc (u, v).

            Return:
                - A dictionary T such that T[(u, v)] gives the fire propagation time from vertex u to vertex v.
                - A dictionary R such that R[(u, v)] gives the harmonic mean of the velocities used to compute T[(u, v)].
        """
        def compute_ROS(u, v, R0, Height, Wind, Speed):
            arc = (u, v)
            # Compute slope
            A_uv = abs(Height[u] - Height[v]) / self.d
            # Compute direction vector between vertices in xy-plane
            n_uv = np.array([v[0] - u[0], v[1] - u[1]])
            n_uv = n_uv / np.linalg.norm(n_uv)
            # Wind speed in the direction of n_uv
            wind_uv = Wind[arc]
            U_uv = Speed[arc] * np.dot(wind_uv, n_uv)  # Signed wind speed        
            # Rothermel rate of spread calculation based on fire type (headfire or backfire)
            slope_type  = ""
            if Height[u] <= Height[v] and U_uv >= 0:
                # Upslope headfire
                R_u = R0[u] * (1 + self.Phi_w(U_uv) + self.Phi_s(A_uv))
                R_v = R0[v] * (1 + self.Phi_w(U_uv) + self.Phi_s(A_uv))
                slope_type =  "Upslope headfire"
            elif Height[u] > Height[v] and U_uv >= 0:
                # Downslope headfire
                R_u = R0[u] * (1 + max(0, self.Phi_w(U_uv) - self.Phi_s(A_uv)))
                R_v = R0[v] * (1 + max(0, self.Phi_w(U_uv) - self.Phi_s(A_uv)))
                slope_type =  "Downslope headfire"
            elif Height[u] <= Height[v] and U_uv < 0:
                # Upslope backfire
                R_u = R0[u] * (1 + max(0, self.Phi_s(A_uv) - self.Phi_w(abs(U_uv))))
                R_v = R0[v] * (1 + max(0, self.Phi_s(A_uv) - self.Phi_w(abs(U_uv))))
                slope_type =  "Upslope backfire"
            else:
                # Downslope backfire
                R_u = R0[u]
                R_v = R0[v]
                slope_type = "Downslope backfire"
            return R_u,  R_v, slope_type
        T = {}
        R = {}
        for arc in A:
            u, v = arc
            R_u, R_v, slope_type = compute_ROS(u, v, R0, Height, Wind, Speed)
            d_uv = np.sqrt((self.d*u[0] - self.d*v[0])**2 + (self.d*u[1] - self.d*v[1])**2 + (Height[u] - Height[v])**2)
            T[arc] =  round(d_uv / (2 * R_u) + d_uv / (2 * R_v), self.precision)
            R[arc] = (2 * R_u * R_v) / (R_u + R_v) 
        return T, R


    def generate_V_and_A(self):
        """
            Return:
                - V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell. 
                - A: a set of arcs. Each arc is represented by a tuple (u, v) where u and v are vertices.
                - coord_to_id: a dictionary such that coord_to_id[(i, j)] gives the id of the cell with xy coordinate (i, j).
                - id_to_coord: a dictionary such that id_to_coord[ID] gives the xy coordinate (i, j) of the cell with id ID.
        """
        V = set()  
        A = set()  
        coord_to_id = {}
        id_to_coord = {}
        vertex_id_counter = 0
        for i in range(self.n):
            for j in range(self.n):
                # Add cell (i, j) to the set of vertices
                V.add((i, j))
                coord_to_id[(i, j)] = vertex_id_counter
                id_to_coord[vertex_id_counter] = (i, j)
                vertex_id_counter += 1
                # Add directed arc to the right neighbor (if it exists)
                for dx, dy in self.neighborhood:
                    if 0 <= i + dx < self.n and 0 <= j + dy < self.n:
                        A.add(((i, j), (i + dx, j + dy)))
        return V, A, coord_to_id, id_to_coord
    

    def generate_graph(self):
        """
            Return:
                - V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell.
                - A: a set of arcs. Each arc is represented by a tuple (u, v) where u and v are vertices.
                - coord_to_id: a dictionary such that coord_to_id[(i, j)] gives the id of the cell with xy coordinate (i, j).
                - id_to_coord: a dictionary such that id_to_coord[ID] gives the xy coordinate (i, j) of the cell with id ID.
                - Height: a dictionary Height such that Height[(i, j)] gives the z coordinate (or the height) of the cell with xy coordinate (i, j).
                - Wind: a dictionary Wind such that Wind[(u, v)] gives the wind direction vector for arc (u, v).
                - Speed: a dictionary Speed such that Speed[(u, v)] gives the wind speed for arc (u, v).
                - R0: a dictionary R0 such that R0[(i, j)] gives the no-wind, no-slope rate of spread inside cell with xy coordinate (i, j).
                - R: a dictionary R such that R[(u, v)] gives the harmonic mean of the velocities used to compute T[(u, v)].
                - T: a dictionary T such that T[(u, v)] gives the fire propagation time from vertex u to vertex v (according to the instance model).
                - ignition_vertex: a tuple (i, j) representing the ignition vertex.
        """
        
        # Generate vertices and arcs
        V, A, coord_to_id, id_to_coord = self.generate_V_and_A()
        # Generate height and wind fields
        R0 = self.generate_R0_field(V)
        Height = self.generate_height_field(V)
        Wind, Speed = self.generate_wind_field(A)
        # Generate propagation times
        T, R = self.generate_propagation_times(A, R0, Height, Wind, Speed)
        # Ignition vertex
        ignition_vertex = int(self.n/2), int(self.n/2)
        return V, A, coord_to_id, id_to_coord, Height, Wind, Speed, R0, R, T, ignition_vertex
    

    def generate_horizon(self, V, fire_arrival_times):
        """
            V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell.
            fire_arrival_times: a dictionary such that fire_arrival_times[v] gives the time at which fire reaches vertex v.
            
            Return:
                - The horizon time H.
        """
        return round(self.Horizon_multiplier * max([fire_arrival_times[u] for u in V if fire_arrival_times[u] < math.inf]), self.precision)


    def generate_resource_distribution(self, V, fire_arrival_times, H):
        """
            V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell.
            fire_arrival_times: a dictionary such that fire_arrival_times[v] gives the time at which fire reaches vertex v.
            H: the time horizon of the instance.

            Return: 
                - resources: a list of dictionaries. The ith entry corresponds to resource i, and resources[i] holds the attributes of the resource.
        """
        def generate_resource():
            """
                Return: a dictionary where the keys are the resource attributes and the values are empty or default values.
            """
            return {"Vb": "NA",
                    "Vp": "NA",
                    "t": None,
                    "c": None,
                    "r": "NA",
                    "z": "NA",
                    "e": "NA",
                    "delta": None}

        fat_dist = [fire_arrival_times[u] for u in V if fire_arrival_times[u] < math.inf]
        q_start, q_end   = (np.quantile(fat_dist, p) for p in [self.Resources_quantile_lb, self.Resources_quantile_ub])
        resources = []
        k = self.Resources_num_resources
        T = self.Resources_decision_points
        res_distribution = [0] * T
        remaining = k
        i = 0
        while remaining > 0:
            res_distribution[i % T] += 1
            remaining -= 1
            i += 1
        rd.shuffle(res_distribution)
        for idx, t in enumerate(np.linspace(q_start, q_end, self.Resources_decision_points).tolist()):
            if res_distribution[idx] == 0:
                continue
            res = generate_resource()
            res["t"] = round(t, self.precision)
            res["c"] = res_distribution[idx]
            delta = round(self.Resources_delay * H, self.precision)
            res["delta"] = delta
            resources.append(res)           
        return resources
    

    def generate_instance(self, inst_name):
        """
            inst_name: the name of the instance to be generated.

            Return: 
                    - V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell.
                    - A: a set of arcs. Each arc is represented by a tuple (u, v) where u and v are vertices.
                    - Z: a dictionary Z such that Z[(i, j)] gives the z coordinate (or the height) of the cell with xy coordinate (i, j).
                    - T: a dictionary T such that T[(u, v)] gives the fire propagation time from vertex u to vertex v.
                    - H: the time horizon.
                    - resources_dist: a list of dictionaries. The ith entry corresponds to resource i, and resources[i] holds the attributes of the resource.
                    - fire_arrival_times: a dictionary such that fire_arrival_times[v] gives the time at which fire reaches vertex v.
                    - ignition_vertex: a tuple (i, j) representing the ignition vertex.
        """
        # Generate the graph
        V, A, coord_to_id, id_to_coord, Z, W, Ws, R0, R, T, ignition_vertex = self.generate_graph()
        # Compute the fire arrival times from the ignition vertex
        fire_arrival_times = dijkstra(V, A, T, ignition_vertex)
        # Generate the horizon and resource distribution
        H = self.generate_horizon(V, fire_arrival_times)
        resources_dist = self.generate_resource_distribution(V, fire_arrival_times, H)
        # Write the instance to a JSON file
        vertex_ids = sorted(id_to_coord.keys())
        data = {}
        data["H"] = H
        data["|V|"] = len(V)
        data["|R|"] = self.Resources_decision_points
        data["I"] = [coord_to_id[(ignition_vertex[0], ignition_vertex[1])]]
        data["Vb"] = "NA"
        data["Vp"] = "NA"
        data["w"] = "NA"
        data["t"] = [r["t"] for r in resources_dist]
        data["c"] = [r["c"] for r in resources_dist]
        data["r"] = "NA"
        data["z"] = "NA"
        data["e"] = "NA"
        data["delta"] = [r["delta"] for r in resources_dist]
        data["arcs"] = []
        for u, v in A:
            u_id = coord_to_id[(u[0], u[1])]
            v_id = coord_to_id[(v[0], v[1])]
            data["arcs"].append([u_id, v_id, T[(u, v)]])
        data["distance"] = {"coordinates": []}
        for vid in vertex_ids:
            v_x, v_y = id_to_coord[vid]
            v_z = Z[(v_x, v_y)]
            data["distance"]["coordinates"].append([self.d * v_x, self.d * v_y, v_z])
        with open(f"{inst_name}.json", 'w') as f:
            json.dump(data, f, indent=4)
        return V, A, Z, T, H, resources_dist, fire_arrival_times, ignition_vertex
                

class Animation:
    
    def __init__(self):
        pass

    @staticmethod
    def run_animation(V, Z, fire_arrival_times, free_burning_time, wind_direction, save_as_gif=False, gif_duration=20, gif_filename="fire_animation.gif", elev_angle=30, azim_angle=45):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract the x, y, z coordinates from V and Z (height values)
        x = np.array([v[0] for v in V])
        y = np.array([v[1] for v in V])
        z = np.array([Z[v] for v in V])

        # Create a 3D bar chart (initially unburned)
        bars = ax.bar3d(x, y, np.zeros_like(z), 1, 1, z, shade=True, color='green')

        # Set up the slider for controlling time
        if not save_as_gif:
            ax_time = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
            slider = Slider(ax_time, 'Time', 0, free_burning_time, valinit=0, valstep=0.1)

        # Wind direction arrow
        wind_origin = [np.mean(x), np.mean(y)]  # Center the arrow at the grid

        # Function to update the bars based on the slider value and add wind arrow
        def update(frame):
            def get_color(v, f):
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
            # Add wind arrow
            ax.quiver(wind_origin[0], wind_origin[1], np.max(z)*1.1,
                      wind_direction[0], wind_direction[1], 1,
                      length=5, color='blue', arrow_length_ratio=0.2)

            # Set the viewing angle to rotate the plot
            #ax.view_init(elev=elev_angle, azim=azim_angle)

        # Call update function when slider value changes
        if not save_as_gif:
            slider.on_changed(update)

        # Initial plot
        update(0)

        # If save_as_gif is True, create the animation and save it as a gif
        if save_as_gif:
            frames_per_second = 2
            total_frames = frames_per_second * gif_duration
            times = np.linspace(0, free_burning_time, total_frames)
            def update_for_gif(frame_idx):
                update(times[frame_idx])
            ani = animation.FuncAnimation(fig, update_for_gif, frames=total_frames, repeat=False)
            ani.save(gif_filename, writer='imagemagick', fps=frames_per_second)
            print(f"GIF saved as {gif_filename}")
        else:
            plt.show()

    def animation(self, V, A, T, Z, ignition_vertex, wind_direction, save_as_gif, gif_duration, gif_filename, elev_angle=50, azim_angle=10):
        fire_arrival_times = dijkstra(V, A, T, ignition_vertex)
        free_burning_time = max(fire_arrival_times.values())
        Animation.run_animation(V, Z, fire_arrival_times, free_burning_time, wind_direction, save_as_gif, gif_duration, gif_filename, elev_angle, azim_angle)


def parse_args():
    parser = argparse.ArgumentParser(description="Instance Generator.")
    
    # Add arguments for grid size and xy step size
    parser.add_argument("--grid", type=str, default="Medium", help="Grid size.")
    parser.add_argument("--slope", type=str, default="Moderate", help="Slope.")
    parser.add_argument("--wind", type=str, default="Light", help="Wind speed.")
    parser.add_argument("--resources_delay", type=str, default="High", help="Magnitude of the delay caused by a resource.")
    parser.add_argument("--num_resources", type=str, default="Moderate", help="Maximum number of protected vertices.")
    parser.add_argument("--resources_dist", type=str, default="Moderate", help="Number of decision points for resource distribution.")
    parser.add_argument("--first_res_time", type=str, default="Early", help="First release time.")
    parser.add_argument("--last_res_time", type=str, default="VeryLate", help="Last release time.")
    parser.add_argument("--seed", type=int, default=123, help="Seed value.")
    parser.add_argument("--save_as_gif", action='store_true', help="Save the animation as a GIF.")
    parser.add_argument("--gif_duration", type=int, default=15, help="Duration of the GIF in seconds.")

    return parser.parse_args()


def main():
    args = parse_args()
    # Generate instance
    instance = Instance(grid=args.grid, wind=args.wind, slope=args.slope, first_release_time=args.first_res_time, last_release_time=args.last_res_time, 
                        num_resources=args.num_resources, resources_dist=args.resources_dist, resources_delay=args.resources_delay, seed=args.seed)
    instance_name = f'{args.grid}_{args.slope}_{args.wind}_{args.resources_delay}_{args.num_resources}_{args.resources_dist}_{args.first_res_time}_{args.last_res_time}_{args.seed}'
    V, A, Z, T, H, resources, fire_arrival_times, ignition_vertex = instance.generate_instance(instance_name)
    # Create animation
    animation = Animation()   
    gif_name = f'{args.grid}_{args.slope}_{args.wind}_{args.seed}' 
    animation.animation(V, A, T, Z, ignition_vertex, instance.Wind_major_direction, args.save_as_gif, args.gif_duration, gif_name + ".gif")


if __name__ == "__main__":
    main()
