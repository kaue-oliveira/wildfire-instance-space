#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import heapq
import noise
import math
from matplotlib.widgets import Slider
import argparse
import random as rd
import json
from pathlib import Path
from matplotlib.collections import LineCollection
import sys
import atexit


class Logger:
    COLORS = {
        'reset': '\x1b[0m',
        'info': '\x1b[34m',      # Blue
        'success': '\x1b[32m',   # Green
        'warn': '\x1b[33m',      # Yellow
        'error': '\x1b[31m',     # Red
        'stat': '\x1b[36m',      # Cyan
        'debug': '\x1b[35m',     # Magenta
    }
    ENABLE_COLOR = sys.stdout.isatty()
    LOG_FILE_HANDLE = None

    @staticmethod
    def _colorize(prefix: str, message: str, kind: str) -> str:
        if Logger.ENABLE_COLOR:
            return f"{Logger.COLORS.get(kind, '')}{prefix}{Logger.COLORS['reset']} {message}"
        return f"{prefix} {message}"

    @staticmethod
    def set_log_file(path: str) -> None:
        try:
            Logger.LOG_FILE_HANDLE = open(path, 'w', encoding='utf-8')
            atexit.register(Logger.close_log_file)
        except Exception as e:
            print(Logger._colorize("[ERROR]", f"Failed to open log file {path}: {e}", 'error'))

    @staticmethod
    def close_log_file() -> None:
        if Logger.LOG_FILE_HANDLE:
            try:
                Logger.LOG_FILE_HANDLE.close()
            finally:
                Logger.LOG_FILE_HANDLE = None

    @staticmethod
    def _emit(prefix: str, message: str, kind: str) -> None:
        print(Logger._colorize(prefix, message, kind))
        if Logger.LOG_FILE_HANDLE:
            try:
                Logger.LOG_FILE_HANDLE.write(f"{prefix} {message}\n")
                Logger.LOG_FILE_HANDLE.flush()
            except Exception:
                pass

    @staticmethod
    def info(message: str) -> None:
        Logger._emit("[INFO]", message, 'info')

    @staticmethod
    def success(message: str) -> None:
        Logger._emit("[OK]", message, 'success')

    @staticmethod
    def warn(message: str) -> None:
        Logger._emit("[WARN]", message, 'warn')

    @staticmethod
    def error(message: str) -> None:
        Logger._emit("[ERROR]", message, 'error')

    @staticmethod
    def debug(message: str) -> None:
        Logger._emit("[DEBUG]", message, 'debug')

    @staticmethod
    def stat(title: str, **metrics: float) -> None:
        parts = [f"{k}={v}" for k, v in metrics.items()]
        message = f"{title}: " + ", ".join(parts)
        Logger._emit("[STAT]", message, 'stat')


def dijkstra(V, A, T, ignition_vertex):
    adj = {v: [] for v in V}
    for (u, v) in A:
        adj[u].append(v)
    dist = {v: float('inf') for v in V}
    dist[ignition_vertex] = 0
    pq = [(0, ignition_vertex)]
    while pq:
        current_dist, u = heapq.heappop(pq)
        if current_dist > dist[u]:
            continue
        for v in adj[u]:
            new_dist = dist[u] + T[(u, v)]
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))
    return dist


class Instance:
    def __init__(self, grid: str, wind: str, slope: str, first_release_time: str, last_release_time: str, num_resources: str, resources_dist: str, resources_delay: str, seed=123, save_plots=False, Nxy=26240, WAF=0.3):
        self.beta = 0.005
        self.sigma = 2000
        self.beta_rel = 1.0
        self.seed = seed
        rd.seed(self.seed)
        self.neighborhood = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        self.grid = grid
        self.wind = wind
        self.slope = slope
        self.first_release_time = first_release_time
        self.last_release_time = last_release_time
        self.num_resources = num_resources
        self.resources_dist = resources_dist
        self.resources_delay = resources_delay
        self.save_plots = save_plots
        self.Nxy = Nxy
        self.WAF = WAF

        # Grid size
        grid_options = {
            "Small": 20,
            "Medium": 30,
            "Large": 40,
            "Huge": 80
        }

        # Capitalize grid if it's not already
        grid = grid.capitalize()
        if grid in grid_options:
            self.n = grid_options[grid]
            self.d = math.ceil(self.Nxy / self.n)
        else:
            print(f'Unknown grid size: {grid}. Options are {list(grid_options.keys())}.')
            exit(0)

        # R0
        self.R0_perlin_scale = self.n
        self.R0_perlin_octaves = 3
        self.R0_perlin_seed = rd.randint(0, 500)
        self.R0_lb = 1
        self.R0_ub = 15

        # Height
        self.Height_perlin_scale = self.n
        self.Height_perlin_octaves = 3
        self.Height_perlin_seed = rd.randint(0, 500)
        self.Height_precision = 3
        self.Height_slope_lb = 0
        self.Height_slope_ub = 0
        # Slope options mapping
        slope_options = {
            "Flat": self.Nxy * math.tan(math.radians(10)),
            "Moderate": self.Nxy * math.tan(math.radians(20)),
            "Steep": self.Nxy * math.tan(math.radians(40))
        }

        slope = slope.capitalize()
        if slope in slope_options:
            self.Height_slope_ub = slope_options[slope]
        else:
            print(f"Unknown slope type: {slope}. Options are {list(slope_options.keys())}.")
            exit(0)

        # Wind
        self.Wind_angle_perlin_scale = self.n
        self.Wind_angle_perlin_octaves = 3
        self.Wind_angle_perlin_seed = rd.randint(0, 500)
        self.Wind_angle_lb = -math.pi/6
        self.Wind_angle_ub = math.pi/6
        self.Wind_speed_perlin_scale =  self.n
        self.Wind_speed_perlin_octaves = 3
        self.Wind_speed_perlin_seed = rd.randint(0, 500)
        self.Wind_major_direction = np.array([0.37139068, 0.92847669])
        
        # Beaufort scale https://en.wikipedia.org/wiki/Beaufort_scale#Modern_scale
        wind_options = {
            "Light": (314.961, 649.606),    # 1.6 m/s to 3.3 m/s
            "Moderate": (1082.68, 1555.12), # 5.5 m/s to 7.9 m/s
            "Strong": (2125.984, 2716.535)  # 10.8 m/s to 13.8 m/s
        }

        if wind in wind_options:
            self.Wind_speed_lb, self.Wind_speed_ub = wind_options[wind]
        else:
            print(f"Unknown wind type: {wind}. Options are {list(wind_options.keys())}.")
            exit(0)
        self.Wind_speed_lb *= self.WAF
        self.Wind_speed_ub *= self.WAF

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
        A = min(A, 1) # Slope cap of 45 degrees (tan(45) = 1)
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


    def normalize_values_to_range(value_map, lb=0, ub=1):
        """
            value_map: a dictionary mapping keys to numerical values.
            lb: lower bound for normalization (default: 0).
            ub: upper bound for normalization (default: 1).

            Return:
                - A dictionary with the same keys but values normalized to [lb, ub] range.
        """
        if not value_map:
            return value_map

        all_values = list(value_map.values())
        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val

        if val_range > 0:
            normalized_map = {key: lb + (value - min_val) / val_range * (ub - lb) for key, value in value_map.items()}
        else:
            # All values are the same, set to lb
            normalized_map = {key: lb for key in value_map.keys()}

        return normalized_map


    def _save_or_show(self, fig, filename_base: str):
        """
            Save figure as PNG if save_plots is True; otherwise discard it.
        """
        if self.save_plots:
            output_path = f"{filename_base}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            Logger.success(f"Saved plot to {output_path}")
        plt.close(fig)


    def reindex_vertex_ids(self, V, coord_to_id):
        """
            Create contiguous vertex IDs 0..|V|-1 after preprocessing/pruning.

            Return:
                - coord_to_id_new: mapping from coord -> new contiguous id
                - id_to_coord_new: mapping from new contiguous id -> coord
        """
        coord_to_id_new = {}
        id_to_coord_new = {}
        for new_id, coord in enumerate(sorted(V)):
            coord_to_id_new[coord] = new_id
            id_to_coord_new[new_id] = coord
        return coord_to_id_new, id_to_coord_new


    def generate_R0_field(self, V):
        """
            V: a set of vertices.

            Return:
                - A dictionary R0 such that R0[(i, j)] gives the no-wind, no-slope rate of spread inside the cell with xy coordinate (i, j).
        """
        R0 = {}
        for v in V:
            x, y = v
            R0[v] = noise.pnoise2(x / self.R0_perlin_scale, y / self.R0_perlin_scale, octaves=self.R0_perlin_octaves, base=self.R0_perlin_seed)
        R0 = Instance.normalize_values_to_range(R0, self.R0_lb, self.R0_ub)
        # Print mean, max, min of R0 field
        Logger.stat("R0 field", mean=float(np.mean(list(R0.values()))), max=float(np.max(list(R0.values()))), min=float(np.min(list(R0.values()))))
        # Plot heatmap of height field
        R0_grid = np.zeros((self.n, self.n), dtype=float)
        for (i, j), r0 in R0.items():
            R0_grid[i, j] = r0
        fig = plt.figure()
        im = plt.imshow(R0_grid.T, origin='lower', cmap='terrain', extent=[0, self.n, 0, self.n], aspect='equal')
        plt.colorbar(im, label='R0 (feet/min)')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"R0 Heatmap (seed: {self.seed}, Perlin seed: {self.R0_perlin_seed})")
        plt.tight_layout()
        self._save_or_show(fig, f"R0_heatmap_seed{self.seed}_perlin{self.R0_perlin_seed}_{self.grid}")
        return R0


    def generate_height_field(self, V):
        """
            V: a set of vertices.

            Return:
                - A dictionary Height such that Height[(i, j)] gives the height of the cell with xy coordinate (i, j).
        """
        # Collect raw Perlin samples
        raw_height_map = {}
        for v in V:
            x, y = v
            perlin_value = noise.pnoise2(x / self.Height_perlin_scale, y / self.Height_perlin_scale, octaves=self.Height_perlin_octaves, base=self.Height_perlin_seed)
            raw_height_map[v] = perlin_value
        # Normalize to [0, Height_slope_ub]
        normalized_height_map = Instance.normalize_values_to_range(raw_height_map, 0, self.Height_slope_ub)
        Height = {v: round(normalized_height_map[v], self.precision) for v in V}

        # Plot heatmap of height field
        height_grid = np.zeros((self.n, self.n), dtype=float)
        for (i, j), h in Height.items():
            height_grid[i, j] = h
        fig = plt.figure()
        im = plt.imshow(height_grid.T, origin='lower', cmap='terrain', extent=[0, self.n, 0, self.n], aspect='equal')
        plt.colorbar(im, label='Height (feet)')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Height Heatmap (Seed: {self.seed}, Perlin seed: {self.Height_perlin_seed}, Slope: {self.slope})")
        plt.tight_layout()
        self._save_or_show(fig, f"Height_heatmap_seed{self.seed}_perlin{self.Height_perlin_seed}_{self.grid}_{self.slope}")
        
        # Print mean, max, min of height field
        Logger.stat("Height field", mean=float(np.mean(list(Height.values()))), max=float(np.max(list(Height.values()))), min=float(np.min(list(Height.values()))))
                
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
        # First pass: collect raw Perlin values for angle and speed
        raw_angle_map = {}
        raw_speed_map = {}
        for arc in A:
            u, v = arc
            u_x, u_y = u
            v_x, v_y = v
            coord_x = (u_x + v_x) / 2
            coord_y = (u_y + v_y) / 2
            # Raw Perlin (angle)
            perlin_angle_raw = noise.pnoise2(coord_x / self.Wind_angle_perlin_scale, coord_y / self.Wind_angle_perlin_scale, octaves=self.Wind_angle_perlin_octaves, base=self.Wind_angle_perlin_seed)
            raw_angle_map[arc] = perlin_angle_raw
            # Raw Perlin (speed)
            perlin_speed_raw = noise.pnoise2(coord_x / self.Wind_speed_perlin_scale, coord_y / self.Wind_speed_perlin_scale, octaves=self.Wind_speed_perlin_octaves, base=self.Wind_speed_perlin_seed)
            raw_speed_map[arc] = perlin_speed_raw
        # Normalize angle to [Wind_angle_lb, Wind_angle_ub] and speed to [Wind_speed_lb, Wind_speed_ub]
        normalized_angle_map = Instance.normalize_values_to_range(raw_angle_map, self.Wind_angle_lb, self.Wind_angle_ub)
        normalized_speed_map = Instance.normalize_values_to_range(raw_speed_map, self.Wind_speed_lb, self.Wind_speed_ub)
        # Second pass: compute directions from normalized angles and assign normalized speeds
        for arc in A:
            perlin_angle = normalized_angle_map[arc]
            rotation_matrix = np.array([[np.cos(perlin_angle), -np.sin(perlin_angle)],
                                        [np.sin(perlin_angle),  np.cos(perlin_angle)]])
            wind_direction = rotation_matrix.dot(w)
            Wind[arc] = wind_direction / np.linalg.norm(wind_direction)
            Speed[arc] = normalized_speed_map[arc]
        # Print mean, max, min of normalized speed map
        Logger.stat("Wind speed map", mean=float(np.mean(list(normalized_speed_map.values()))), max=float(np.max(list(normalized_speed_map.values()))), min=float(np.min(list(normalized_speed_map.values()))))
        # Print mean, max, min of normalized angle map
        Logger.stat("Wind angle map", mean=float(np.mean(list(normalized_angle_map.values()))), max=float(np.max(list(normalized_angle_map.values()))), min=float(np.min(list(normalized_angle_map.values()))))
        
        # Plot wind direction vectors (Wind[(u, v)]) as arrows
        fig = plt.figure()
        for arc in A:
            u, v = arc
            wind_vec = Wind[arc]
            # Plot arrow at the midpoint of u and v, in the direction of Wind[(u, v)]
            mid_x = (u[0] + v[0]) / 2
            mid_y = (u[1] + v[1]) / 2
            arrow_scale = 1.0  # scale for visibility
            plt.arrow(mid_x, mid_y, wind_vec[0]*arrow_scale, wind_vec[1]*arrow_scale,
                      head_width=0.2, head_length=0.3, fc='b', ec='b', alpha=0.7)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"Wind vectors (Seed: {self.seed}, Angle seed: {self.Wind_angle_perlin_seed}, Wind: {self.wind})")
        plt.axis('equal')
        plt.tight_layout()
        self._save_or_show(fig, f"Wind_vectors_seed{self.seed}_angle{self.Wind_angle_perlin_seed}_{self.grid}_{self.wind}")
        
        # Visualize per-arc wind speed using colored segments
        segments = []
        colors = []
        for (u, v), speed in normalized_speed_map.items():
            #x0, y0 = (u[0] + v[0]) / 2, (u[1] + v[1]) / 2
            x0, y0 = u[0], u[1]
            x1, y1 = v[0], v[1]
            segments.append([[x0, y0], [x1, y1]])
            colors.append(speed)
        lc = LineCollection(segments, cmap='viridis', linewidths=1.5)
        lc.set_array(np.array(colors))
        fig, ax = plt.subplots()
        ax.add_collection(lc)
        ax.set_xlim(0, self.n)
        ax.set_ylim(0, self.n)
        ax.set_aspect('equal', adjustable='box')
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Wind speed (feet/min)')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Per-arc wind speeds (Seed: {self.seed}, Speed seed: {self.Wind_speed_perlin_seed}, Wind: {self.wind})")
        plt.tight_layout()
        self._save_or_show(fig, f"Wind_speed_segments_seed{self.seed}_speed{self.Wind_speed_perlin_seed}_{self.grid}_{self.wind}")

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
        
        
        # For each arc, compute the slope angle and print the mean, max, min of the slope angles (in degrees)
        slope_angles = []
        for arc in A:
            u, v = arc
            slope_angle = np.arctan2(abs(Height[v] - Height[u]), self.d)
            slope_angles.append(slope_angle * 180 / math.pi)
        Logger.stat("Slope angles (deg)", mean=float(np.mean(slope_angles)), max=float(np.max(slope_angles)), min=float(np.min(slope_angles)))
       
        return V, A, coord_to_id, id_to_coord, Height, Wind, Speed, R0, R, T, ignition_vertex


    def generate_horizon(self, V, fire_arrival_times):
        """
            V: a set of vertices. Each vertex is represented by a tuple (i, j) where i and j are the xy coordinates of the cell.
            fire_arrival_times: a dictionary such that fire_arrival_times[v] gives the time at which fire reaches vertex v.

            Return:
                - The horizon time H, used as the pruning threshold. H is computed as:
                  min(max(t90, 24h), 48h) and then raised if necessary to ensure that at
                  least 70% of vertices are kept, i.e., H := max(H, q70). Here, t90 is the
                  90th percentile and q70 is the 70th percentile of finite fire arrival times.
        """
        finite_times = [fire_arrival_times[u] for u in V if fire_arrival_times[u] < math.inf]
        if finite_times:
            q100 = float(np.quantile(finite_times, 1))
            q70 = float(np.quantile(finite_times, 0.70))
        else:
            q100 = 0.0
            q70 = 0.0
        lower_bound = 24 * 60
        upper_bound = 48 * 60
        base_threshold = min(max(q100, lower_bound), upper_bound)
        # Ensure at least 70% of vertices are kept
        threshold_minutes = max(base_threshold, q70)
        
        Logger.stat("Horizon thresholds (min)", q100=q100, q70=q70, lower_bound=lower_bound, upper_bound=upper_bound, H=threshold_minutes)
        return round(threshold_minutes, self.precision)


    def prune_by_time_threshold(self, V, A, coord_to_id, id_to_coord, Height, Wind, Speed, R0, R, T, fire_arrival_times, H):
        """
            Prune vertices and arcs based on the provided horizon H (in minutes).
            H is typically computed as min(max(t90, 24h), 48h) and then adjusted to
            keep at least 70% of vertices by setting H := max(H, q70), where t90 and q70
            are the 90th and 70th percentiles of finite fire arrival times, respectively.

            Return:
                - Pruned V, A, coord_to_id, id_to_coord, Height, Wind, Speed, R0, R, T
        """
        threshold_minutes = H

        V_pruned = [v for v in V if fire_arrival_times[v] <= threshold_minutes]
        A_pruned = [arc for arc in A if fire_arrival_times[arc[0]] <= threshold_minutes and fire_arrival_times[arc[1]] <= threshold_minutes]
        coord_to_id_pruned = {coord: id for coord, id in coord_to_id.items() if fire_arrival_times[coord] <= threshold_minutes}
        id_to_coord_pruned = {id: coord for id, coord in id_to_coord.items() if fire_arrival_times[coord] <= threshold_minutes}
        Height_pruned = {coord: height for coord, height in Height.items() if fire_arrival_times[coord] <= threshold_minutes}
        Wind_pruned = {arc: wind for arc, wind in Wind.items() if fire_arrival_times[arc[0]] <= threshold_minutes and fire_arrival_times[arc[1]] <= threshold_minutes}
        Speed_pruned = {arc: speed for arc, speed in Speed.items() if fire_arrival_times[arc[0]] <= threshold_minutes and fire_arrival_times[arc[1]] <= threshold_minutes}
        R0_pruned = {coord: r0 for coord, r0 in R0.items() if fire_arrival_times[coord] <= threshold_minutes}
        R_pruned = {arc: r for arc, r in R.items() if fire_arrival_times[arc[0]] <= threshold_minutes and fire_arrival_times[arc[1]] <= threshold_minutes}
        T_pruned = {arc: t for arc, t in T.items() if fire_arrival_times[arc[0]] <= threshold_minutes and fire_arrival_times[arc[1]] <= threshold_minutes}

        return V_pruned, A_pruned, coord_to_id_pruned, id_to_coord_pruned, Height_pruned, Wind_pruned, Speed_pruned, R0_pruned, R_pruned, T_pruned


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
        # Generate the horizon (threshold minutes)
        H = self.generate_horizon(V, fire_arrival_times)
        # Prune graph based on time threshold H
        V, A, coord_to_id, id_to_coord, Z, W, Ws, R0, R, T = self.prune_by_time_threshold(
            V, A, coord_to_id, id_to_coord, Z, W, Ws, R0, R, T, fire_arrival_times, H
        )
        # Reindex vertex IDs to be contiguous 0..|V|-1
        coord_to_id, id_to_coord = self.reindex_vertex_ids(V, coord_to_id)
        # Build arc list in terms of reindexed vertex IDs
        A_ids = sorted([(coord_to_id[u], coord_to_id[v]) for (u, v) in A])
        Logger.info(f'Number of vertices after pruning: {len(V)}')
        # Generate the resource distribution
        resources_dist = self.generate_resource_distribution(V, fire_arrival_times, H)
        # Write the instance to a JSON file
        vertex_ids = sorted(id_to_coord.keys())
        data = {
            "H": H,
            "|V|": len(V),
            "|R|": self.Resources_decision_points,
            "I": [coord_to_id[(ignition_vertex[0], ignition_vertex[1])]],
            "Vb": "NA",
            "Vp": "NA",
            "w": "NA",
            "t": [r["t"] for r in resources_dist],
            "c": [r["c"] for r in resources_dist],
            "r": "NA",
            "z": "NA",
            "e": "NA",
            "delta": [r["delta"] for r in resources_dist],
            "arcs": []
        }
        for u_id, v_id in A_ids:
            u_coord = id_to_coord[u_id]
            v_coord = id_to_coord[v_id]
            data["arcs"].append([u_id, v_id, T[(u_coord, v_coord)]])
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
    def run_animation(V, Z, fire_arrival_times, free_burning_time, wind_direction, elev_angle, azim_angle):

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract the x, y, z coordinates from V and Z (height values)
        x = np.array([v[0] for v in V])
        y = np.array([v[1] for v in V])
        z = np.array([Z[v] for v in V])

        # Create a 3D bar chart (initially unburned)
        bars = ax.bar3d(x, y, np.zeros_like(z), 1, 1, z, shade=True, color='green')

        # Set up the slider for controlling time
        ax_time = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_time, 'Time', 0, free_burning_time, valinit=0, valstep=0.1)

        # Wind direction arrow
        wind_origin = [np.mean(x), np.mean(y)]  # Center the arrow at the grid

        # Set initial viewing angle
        ax.view_init(elev=elev_angle, azim=azim_angle)

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

        # Call update function when slider value changes
        slider.on_changed(update)

        # Initial plot
        update(0)

        plt.show()

    def animation(self, V, A, T, Z, ignition_vertex, wind_direction, elev_angle=40, azim_angle=195):
        """
        Display an interactive 3D visualization of fire propagation with a time slider.

        Args:
            V: Set of vertices
            A: Set of arcs
            T: Fire propagation times
            Z: Height field
            ignition_vertex: Starting point of fire
            wind_direction: Wind direction vector
            elev_angle: Elevation angle for 3D view (default: 40)
            azim_angle: Azimuth angle for 3D view (default: 195)
        """
        fire_arrival_times = dijkstra(V, A, T, ignition_vertex)
        free_burning_time = max(fire_arrival_times.values())
        Animation.run_animation(V, Z, fire_arrival_times, free_burning_time, wind_direction, elev_angle, azim_angle)


def parse_args():
    parser = argparse.ArgumentParser(description="Instance Generator.")

    # Add arguments for grid size and xy step size
    parser.add_argument("--grid", type=str, default="Medium", help="Grid size.")
    parser.add_argument("--slope", type=str, default="Moderate", help="Slope.")
    parser.add_argument("--wind", type=str, default="Light", help="Wind.")
    parser.add_argument("--resources_delay", type=str, default="High", help="Delay caused by a resource.")
    parser.add_argument("--num_resources", type=str, default="Moderate", help="Number of resources.")
    parser.add_argument("--decision_points", type=str, default="Moderate", help="Number of decision points.")
    parser.add_argument("--first_res_time", type=str, default="Early", help="First release time.")
    parser.add_argument("--last_res_time", type=str, default="VeryLate", help="Last release time.")
    parser.add_argument("--seed", type=int, default=123, help="Seed value.")
    parser.add_argument("--show_animation", action='store_true', help="Display interactive 3D visualization with time slider.")
    parser.add_argument("--save_plots", action='store_true', help="Save plots as PNG instead of displaying them.")
    parser.add_argument("--Nxy", type=float, default=26240, help="XY step scaling parameter.")
    parser.add_argument("--WAF", type=float, default=0.3, help="Wind Adjustment Factor (scales wind speeds).")

    return parser.parse_args()


def main():
    args = parse_args()
    # Generate instance
    instance = Instance(grid=args.grid, wind=args.wind, slope=args.slope, first_release_time=args.first_res_time, last_release_time=args.last_res_time,
                        num_resources=args.num_resources, resources_dist=args.decision_points, resources_delay=args.resources_delay, seed=args.seed, save_plots=args.save_plots, Nxy=args.Nxy, WAF=args.WAF)
    instance_name = f'{args.grid}_{args.slope}_{args.wind}_{args.resources_delay}_{args.num_resources}_{args.decision_points}_{args.first_res_time}_{args.last_res_time}_{args.seed}'
    # Initialize file logging to <instance_name>.log
    Logger.set_log_file(f"{instance_name}.log")
    V, A, Z, T, H, resources, fire_arrival_times, ignition_vertex = instance.generate_instance(instance_name)
    # Show interactive animation if requested
    if args.show_animation:
        anim = Animation()
        anim.animation(V, A, T, Z, ignition_vertex, instance.Wind_major_direction)


if __name__ == "__main__":
    main()