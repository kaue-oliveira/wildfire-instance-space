#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <ranges>
#include <regex>
#include <string>
#include <vector>
using namespace std;
namespace views = std::views;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <boost/multi_array.hpp>
using namespace boost;

#include "common.hpp"
#include "digraph.hpp"
#include "heap.hpp"

struct COORDINATE {
  int x, y;

  COORDINATE(int _x, int _y) : x(_x), y(_y) {}

  string to_string() const { return "(" + std::to_string(x) + ", " + std::to_string(y) + ")"; }

  bool operator==(const COORDINATE &other) const { return x == other.x && y == other.y; }

  bool operator<(const COORDINATE &other) const {
    return x < other.x || (x == other.x && y < other.y);
  }
};

struct COORDINATE_3D : public COORDINATE {
  int z;

  COORDINATE_3D(int _x, int _y, int _z) : COORDINATE(_x, _y), z(_z) {}

  string to_string() const {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
  }

  bool operator==(const COORDINATE_3D &other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator<(const COORDINATE_3D &other) const {
    return x < other.x || (x == other.x && y < other.y) ||
           (x == other.x && y == other.y && z < other.z);
  }
};

class Instance {
 private:
  string instance_id;
  VERTEXID num_vertices;
  DISTANCE xy_dist;
  TIME H;              // Optimization horizon
  vector<VERTEXID> I;  // Ignition vertices
  vector<VERTEXID> S;  // Shelter vertices
  vector<RESOURCE> R;  // Resources
  vector<TIME> delta;  // Delay values
  vector<TIME> t;      // Time at which resource i ∈ R is available
  vector<CAPACITY> c;  // Number of vertices that can be protected by resource i ∈ R
  vector<VERTEXID> Vf;
  vector<vector<VERTEXID>> Vb;  // Set of vertices that can be the base location of resource i ∈ R
  vector<vector<VERTEXID>> Vp;  // Set of vertices that can be the protected by resource i ∈ R
  vector<DISTANCE> r;           // Range of resource i ∈ R
  vector<VALUE> w;              // Value of vertex v ∈ V
  vector<TIME> zb;              // Safety time for the base location of resource i ∈ R
  vector<TIME> zp;              // Safety time for the vertices protected by resource i ∈ R
  vector<TIME> db;              // Safety distance for the base location of resource i ∈ R
  vector<TIME> dp;              // Safety distance for the vertices protected by resource i ∈ R
  vector<TIME> e;               // Expiration time for resource i ∈ R
  vector<COORDINATE_3D> coordinates;  // Vertex coordinates
  Digraph G;                          // Graph
  map<COORDINATE, VERTEXID> coord_to_id;
  vector<TIME> delayID_2_delayValue;
  vector<DELAYID> resID_2_delayID;
  bool no_Vp;
  bool no_Vb;
  bool no_value;
  bool basic_problem;

  DISTANCE manhattan2D(COORDINATE a, COORDINATE b) { return abs(a.x - b.x) + abs(a.y - b.y); }

  DISTANCE manhattan3D(COORDINATE_3D a, COORDINATE_3D b) {
    return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
  }

  DISTANCE euclidean2D(COORDINATE a, COORDINATE b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  DISTANCE euclidean3D(COORDINATE_3D a, COORDINATE_3D b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
  }

  // Methods to load various components of the instance from JSON
  void load_vertices_count(const json &instance) { num_vertices = instance["|V|"].get<VERTEXID>(); }

  void load_optimization_horizon(const json &instance) { H = instance["H"].get<TIME>(); }

  void load_ignition_vertices(const json &instance) { I = instance["I"].get<vector<VERTEXID>>(); }

  void load_shelter_vertices(const json &instance) {
    if (!instance.contains("S"))
      S = vector<VERTEXID>();
    else
      S = instance["S"].get<vector<VERTEXID>>();
  }

  void load_resources(const json &instance) {
    for (RESOURCE r = 0; r < instance["|R|"].get<RESOURCE>(); r++) R.push_back(r);
  }

  void load_base_locations(const json &instance) {
    if (instance["Vb"].is_string())
      no_Vb = true;
    else {
      for (const auto &S : instance["Vb"]) Vb.push_back(S.get<vector<VERTEXID>>());
      no_Vb = false;
    }
  }

  void load_protected_vertices(const json &instance) {
    if (instance["Vp"].is_string())
      no_Vp = true;
    else {
      for (const auto &S : instance["Vp"]) Vp.push_back(S.get<vector<VERTEXID>>());
      no_Vp = false;
    }
  }

  void load_vertex_values(const json &instance) {
    if (instance["w"].is_string()) {
      no_value = true;
      w = vector<VALUE>(num_vertices, VALUE(1));
    } else {
      w = instance["w"].get<vector<VALUE>>();
      no_value = false;
    }
  }

  void load_release_times(const json &instance) { t = instance["t"].get<vector<TIME>>(); }

  void load_resource_capacity(const json &instance) { c = instance["c"].get<vector<CAPACITY>>(); }

  void load_safety_times(const json &instance) {
    // Check if "zb" exists
    if (instance.contains("zb")) {
      if (instance["zb"].is_string()) {
        zb.resize(R.size(), TIME(0));
      } else {
        for (RESOURCE i : R) {
          if (instance["zb"][i].is_string())
            zb.push_back(TIME(0));
          else
            zb.push_back(instance["zb"][i].get<TIME>());
        }
      }
    } else {
      // If "zb" does not exist, fill zb with 0's
      zb.resize(R.size(), TIME(0));
    }

    // Check if "zp" exists
    if (instance.contains("zp")) {
      if (instance["zp"].is_string()) {
        zp.resize(R.size(), TIME(0));
      } else {
        for (RESOURCE i : R) {
          if (instance["zp"][i].is_string())
            zp.push_back(TIME(0));
          else
            zp.push_back(instance["zp"][i].get<TIME>());
        }
      }
    } else {
      // If "zp" does not exist, fill zp with 0's
      zp.resize(R.size(), TIME(0));
    }
  }

  void load_safety_distances(const json &instance) {
    // Check if "db" exists
    if (instance.contains("db")) {
      if (instance["db"].is_string()) {
        db.resize(R.size(), DISTANCE(0));
      } else {
        for (RESOURCE i : R) {
          if (instance["db"][i].is_string())
            db.push_back(INF_DISTANCE);
          else
            db.push_back(instance["db"][i].get<DISTANCE>());
        }
      }
    } else {
      // If "db" does not exist, fill db with 0's
      db.resize(R.size(), DISTANCE(0));
    }

    // Check if "dp" exists
    if (instance.contains("dp")) {
      if (instance["dp"].is_string()) {
        dp.resize(R.size(), DISTANCE(0));
      } else {
        for (RESOURCE i : R) {
          if (instance["dp"][i].is_string())
            dp.push_back(INF_DISTANCE);
          else
            dp.push_back(instance["dp"][i].get<DISTANCE>());
        }
      }
    } else {
      // If "dp" does not exist, fill dp with 0's
      dp.resize(R.size(), DISTANCE(0));
    }
  }

  void load_expiration_times(const json &instance) {
    // Check if "e" exists
    if (instance.contains("e")) {
      if (instance["e"].is_string()) {
        e.resize(R.size(), INF_TIME);
      } else {
        for (RESOURCE i : R) {
          if (instance["e"][i].is_string())
            e.push_back(INF_TIME);
          else
            e.push_back(instance["e"][i].get<TIME>());
        }
      }
    } else {
      // If "e" does not exist, fill e with INF_TIME
      e.resize(R.size(), INF_TIME);
    }
  }

  void load_range(const json &instance) {
    // Check if "r" exists
    if (instance.contains("r")) {
      if (instance["r"].is_string()) {
        // If "r" is a string, fill r with INF_DISTANCE
        r.resize(R.size(), INF_DISTANCE);
      } else {
        // If "r" is an array, process each element
        for (RESOURCE i : R) {
          if (instance["r"][i].is_string())
            r.push_back(INF_DISTANCE);
          else
            r.push_back(instance["r"][i].get<DISTANCE>());
        }
      }
    } else {
      // If "r" does not exist, fill r with INF_DISTANCE
      r.resize(R.size(), INF_DISTANCE);
    }
  }

  void load_delay_values(const json &instance) { delta = instance["delta"].get<vector<TIME>>(); }

  void load_coordinates(const json &instance) {
    VERTEXID i = 0;
    for (const auto &coord : instance["distance"]["coordinates"]) {
      coord_to_id[{coord[0], coord[1]}] = i++;
      coordinates.emplace_back(coord[0], coord[1], coord[2]);
    }
  }

  void load_arcs(const json &instance) {
    G = Digraph(num_vertices);
    for (const auto &arc : instance["arcs"]) {
      assert(arc.size() == 3);                                 // Check that the arc has 3 elements
      assert(arc[0] < num_vertices && arc[1] < num_vertices);  // Check that the vertices are valid
      assert(arc[2] > 0);                                      // Check that the cost is positive
      G.add_arc(arc[0], arc[1], arc[2]);
    }
  }

  /*
      - Since there is a finite number of resources, each with an associated
     delay value, there's a finite number of distinct delay values.
      - We create a mapping between these delay values and a unique ID (DELAYID)
     for each resource.
      - It's possible to show that a solution to the problem is completely
     determined by the delay assigned to each vertex.
      - We represent solutions by a vector of characters (DELAYID) of size |V|,
     where each character corresponds to the delay assigned to the vertex.
      - There's a special character NULL_DELAY, which indicates that the vertex
     is not protected by any resource.
  */
  void create_delay_mappings() {
    set<TIME> delay_values(delta.begin(), delta.end());
    map<TIME, DELAYID> delayValue_2_delayID;
    delayID_2_delayValue = vector<TIME>(delay_values.size() + 1);
    delayID_2_delayValue[0] = TIME(0);
    for (unsigned idx = 1; idx < delayID_2_delayValue.size(); idx++) {
      TIME delayValue = *next(delay_values.begin(), idx - 1);
      delayID_2_delayValue[idx] = delayValue;
      delayValue_2_delayID[delayValue] = idx;
    }
    resID_2_delayID = vector<DELAYID>(R.size(), NULL_DELAY);
    for (RESOURCE i : R) resID_2_delayID[i] = delayValue_2_delayID[delta[i]];
  }

  void sort_data_structures() {
    for (RESOURCE i : R) {
      if (!no_Vp) sort(Vp[i].begin(), Vp[i].end());
      if (!no_Vb) sort(Vb[i].begin(), Vb[i].end());
    }
    sort(I.begin(), I.end());
    sort(S.begin(), S.end());
    sort(Vf.begin(), Vf.end());
  }

  void compute_flammable_vertices() {
    MyHeap heap(num_vertices);
    vector<TIME> fire_arrival_time(num_vertices, INF_TIME);
    for (VERTEXID i : I) {
      heap.insertElement(i, 0);
      fire_arrival_time[i] = 0;
    }
    while (!heap.empty()) {
      VERTEXID u = heap.findAndDeleteMinElement();
      TIME a_u = fire_arrival_time[u];
      Vf.push_back(u);
      for (auto &[v, t_uv] : G.get_outgoing_arcs(u)) {
        if (ff(fire_arrival_time[v]) > ff(a_u + t_uv)) {
          heap.adjustHeap(v, a_u + t_uv);
          fire_arrival_time[v] = a_u + t_uv;
        }
      }
    }
  }

  // Check whether the distance between vertices in the xy plane is always the
  // same.
  DISTANCE check_xy_plane_uniformity() {
    DISTANCE d = -1;
    for (VERTEXID u : get_V()) {
      for (auto &[v, t_uv] : G.get_outgoing_arcs(u)) {
        DISTANCE dist = euclidean2D(coordinates[u], coordinates[v]);
        if (d == -1) {
          d = dist;
        } else if (dist != d) {
          cout << "Distance between vertices " << u << " and " << v << " is not uniform." << endl;
          cout << "	Distance: " << dist << endl;
          cout << "	Expected: " << d << endl;
          cout << "	Coordinates: " << coordinates[u].to_string() << ", "
               << coordinates[v].to_string() << endl;
          return INF_DISTANCE;
        }
      }
    }
    return d;
  }

 public:
  static const DELAYID NULL_DELAY = 0;

  Instance(const string &instance_file) {
    ifstream f(instance_file);
    if (!f.good()) {
      cerr << "Could not open instance specification file." << endl;
      exit(1);
    }
    json instance = json::parse(f);
    f.close();
    string filename = instance_file.substr(instance_file.find_last_of("/") + 1);
    string::size_type const p(filename.find_last_of('.'));
    instance_id = filename.substr(0, p);

    // 1. Number of vertices
    load_vertices_count(instance);
    // 2. Optimization horizon
    load_optimization_horizon(instance);
    // 3. Ignition vertices
    load_ignition_vertices(instance);
    // 4. Shelter vertices (not relevant for the current problem)
    load_shelter_vertices(instance);
    // 5. Resources
    load_resources(instance);
    // 6. Base locations
    load_base_locations(instance);
    // 7. Protected vertices
    load_protected_vertices(instance);
    // 8. Vertex values
    load_vertex_values(instance);
    // 9. Release times
    load_release_times(instance);
    // 10. Resource capacity
    load_resource_capacity(instance);
    // 11. Safety times
    load_safety_times(instance);
    // 12. Safety distances
    load_safety_distances(instance);
    // 13. Expiration times
    load_expiration_times(instance);
    // 14. Range
    load_range(instance);
    // 15. Delay values
    load_delay_values(instance);
    // 16. Coordinates
    load_coordinates(instance);
    // 17. Arcs
    load_arcs(instance);

    // Create a mapping between resources and delays
    create_delay_mappings();
    // Compute additional data structures
    compute_flammable_vertices();
    xy_dist = check_xy_plane_uniformity();
    // Sort data structures to improve searching
    sort_data_structures();
    basic_problem = is_basic_problem();
  }

  bool is_basic_problem() {
    // Check if all decision points are distinct, i.e., vector t has no repeated
    // elements
    set<TIME> t_set(t.begin(), t.end());
    if (t_set.size() != t.size()) return false;
    // Check that all delays are the same
    set<TIME> delta_set(delta.begin(), delta.end());
    if (delta_set.size() != 1) return false;
    // Check that Vb and Vp are the same for all resources
    if (!no_Vb || !no_Vp) return false;
    // Check that all resources have an infinite range, infinite expiration
    // time, and zero safety time
    for (RESOURCE i : R) {
      if (r[i] != INF_DISTANCE || e[i] != INF_TIME || zb[i] != TIME(0) || zp[i] != TIME(0))
        return false;
    }
    // Check that all resources have a value of 1
    for (VERTEXID v : get_V()) {
      if (w[v] != VALUE(1)) return false;
    }
    // Check that the graph is a square grid
    DISTANCE d = check_xy_plane_uniformity();
    if (d == INF_DISTANCE) return false;
    return true;
  }

  ranges::iota_view<VERTEXID, VERTEXID> get_V() {
    return ranges::iota_view<VERTEXID, VERTEXID>{0, num_vertices};
  }

  VERTEXID get_vertex_ID(COORDINATE coord) {
    if (coord_to_id.find(coord) == coord_to_id.end())
      return INF_VERTEXID;
    else
      return coord_to_id[coord];
  }

  auto get_outgoing_arcs(VERTEXID v) { return G.get_outgoing_arcs(v); }

  auto get_incoming_arcs(VERTEXID v) { return G.get_incoming_arcs(v); }

  inline TIME get_arc_cost(VERTEXID u, VERTEXID v) { return G.get_arc_cost(u, v); }

  const vector<VERTEXID> &get_Vf() { return Vf; }

  const vector<VERTEXID> &get_Vb(RESOURCE i) {
    if (no_Vb)
      return Vf;
    else
      return Vb[i];
  }

  const vector<VERTEXID> &get_Vp(RESOURCE i) {
    if (no_Vp)
      return Vf;
    else
      return Vp[i];
  }

  const vector<VERTEXID> &get_I() { return I; }

  const vector<VERTEXID> &get_S() { return S; }

  const vector<RESOURCE> &get_R() { return R; }

  string get_vertex_signature(VERTEXID u) { return coordinates[u].to_string(); }

  DISTANCE distance(VERTEXID u, VERTEXID v) {
    auto coord_u = coordinates[u];
    auto coord_v = coordinates[v];
    return euclidean3D(coord_u, coord_v);
  }

  bool inVb(VERTEXID v, RESOURCE i) {
    if (no_Vb)
      return binary_search(Vf.begin(), Vf.end(), v);
    else
      return binary_search(Vb[i].begin(), Vb[i].end(), v);
  }

  bool inVp(VERTEXID v, RESOURCE i) {
    if (no_Vp)
      return binary_search(Vf.begin(), Vf.end(), v);
    else
      return binary_search(Vp[i].begin(), Vp[i].end(), v);
  }

  bool inVf(VERTEXID v) { return binary_search(Vf.begin(), Vf.end(), v); }

  bool inS(VERTEXID v) { return binary_search(S.begin(), S.end(), v); }

  bool inI(VERTEXID v) { return binary_search(I.begin(), I.end(), v); }

  inline string get_instance_id() { return instance_id; }

  inline VERTEXID get_num_vertices() { return num_vertices; }

  inline RESOURCE get_num_resources() { return R.size(); }

  inline DISTANCE get_range(RESOURCE i) { return r[i]; }

  inline TIME get_base_safety_time(RESOURCE i) { return zb[i]; }

  inline TIME get_protection_safety_time(RESOURCE i) { return zp[i]; }

  // For backward compatibility
  inline TIME get_safety_time(RESOURCE i) {
    return zb[i];  // Default to base safety time
  }

  inline DISTANCE get_base_safety_distance(RESOURCE i) { return db[i]; }

  inline DISTANCE get_protection_safety_distance(RESOURCE i) { return dp[i]; }

  inline TIME get_expiration_time(RESOURCE i) { return e[i]; }

  inline TIME get_Delta() {
    assert(basic_problem);
    return delta[0];
  }

  inline VALUE get_value(VERTEXID v) { return w[v]; }

  inline TIME get_resource_delay(RESOURCE i) { return delta[i]; }

  inline TIME get_release_time(RESOURCE i) { return t[i]; }

  inline CAPACITY get_capacity(RESOURCE i) { return c[i]; }

  inline TIME get_optimization_horizon() { return H; }

  inline DISTANCE get_xy_distance() { return xy_dist; }

  inline COORDINATE_3D get_coordinate(VERTEXID v) { return coordinates[v]; }

  inline bool coordinate_exists(COORDINATE coord) {
    return coord_to_id.find(coord) != coord_to_id.end();
  }

  pair<int, int> get_x_range() {
    int min_x =
        (*min_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.x < b.x; }))
            .x;
    int max_x =
        (*max_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.x < b.x; }))
            .x;
    return {min_x, max_x};
  }

  pair<int, int> get_y_range() {
    int min_y =
        (*min_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.y < b.y; }))
            .y;
    int max_y =
        (*max_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.y < b.y; }))
            .y;
    return {min_y, max_y};
  }

  pair<int, int> get_z_range() {
    int min_z =
        (*min_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.z < b.z; }))
            .z;
    int max_z =
        (*max_element(coordinates.begin(), coordinates.end(),
                      [](const COORDINATE_3D &a, const COORDINATE_3D &b) { return a.z < b.z; }))
            .z;
    return {min_z, max_z};
  }

  inline DELAYID get_delay_ID(RESOURCE i) { return resID_2_delayID[i]; }

  inline TIME get_delay_value(DELAYID d) { return delayID_2_delayValue[d]; }

  // The methods below are only used by Yen's algorithm to compute the kth
  // shortest path.
  inline void restore_vertex(VERTEXID u) { G.restore_vertex(u); }

  inline void restore_arc(VERTEXID u, VERTEXID v) { G.restore_arc(u, v); }

  inline void block_vertex(VERTEXID u) { G.block_vertex(u); }

  inline void block_arc(VERTEXID u, VERTEXID v) { G.block_arc(u, v); }

  inline void restore_vertices() { G.restore_vertices(); }

  inline void restore_arcs() { G.restore_arcs(); }
};
