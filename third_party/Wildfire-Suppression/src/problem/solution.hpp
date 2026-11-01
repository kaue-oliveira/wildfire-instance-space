#pragma once

#include <cmath>
#include <fstream>
#include <limits>
#include <list>
#include <map>
#include <ranges>
#include <string>
#include <vector>
using namespace std;
namespace views = std::views;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "heap.hpp"
#include "instance.hpp"

class Solution {
 public:
  Instance &I;
  // Fire propagation
  vector<TIME> fire_arrival_time;
  vector<VERTEXID> predecessor;
  // Resource allocation
  string fingerprint;              // A string indicating the delay associated with each
                                   // vertex (a solution can be fully characterized by the
                                   // amount of delay added to each vertex). Check
                                   // Instance::create_delay_mappings() for a better
                                   // understanding of this.
  vector<TIME> deployment_time;    // The time at which each resource is deployed
  vector<VERTEXID> base_location;  // The base location of each resource
  vector<list<VERTEXID>> protected_vertices;  // The vertices protected by each resource
  bool partial_solution;                      // Indicates whether the solution is partial (i.e., it
                                              // does not contain the fire propagation information)
                                              // or not
  // Statistics
  VALUE objv;               // Objective value
  VALUE lb;                 // Lower bound
  VERTEXID burnt_vertices;  // Number of burnt vertices at H
  TIME time_to_survival;    // Time to survival
  unsigned timestamp;       // Timestamp of the solution
  unsigned iter;            // Iteration number of the solution (only updated by some
                            // algorithms)

  Solution(Instance &_I, bool initialize = false, bool initialize_feasible = false)
      : I(_I),
        objv(INF_VALUE),
        lb(0),
        burnt_vertices(INF_VERTEXID),
        time_to_survival(INF_TIME),
        timestamp(0),
        iter(0) {
    partial_solution = true;
    if (initialize)  // Allocate space
    {
      partial_solution = false;
      fire_arrival_time = vector<TIME>(I.get_num_vertices(), INF_TIME);
      predecessor = vector<VERTEXID>(I.get_num_vertices(), INF_VERTEXID);
      fingerprint = string(I.get_num_vertices(), Instance::NULL_DELAY);
      deployment_time = vector<TIME>(I.get_num_resources(), INF_TIME);
      base_location = vector<VERTEXID>(I.get_num_resources(), INF_VERTEXID);
      protected_vertices = vector<list<VERTEXID>>(I.get_num_resources());
      if (initialize_feasible)  // Initialize a feasible solution (empty
                                // allocation)
      {
        Solution::compute_shortest_path_forest(*this);
      }
    }
  }

  Solution(const Solution &other) : I(other.I) {
    fire_arrival_time = other.fire_arrival_time;
    predecessor = other.predecessor;
    fingerprint = other.fingerprint;
    deployment_time = other.deployment_time;
    base_location = other.base_location;
    protected_vertices = other.protected_vertices;
    objv = other.objv;
    time_to_survival = other.time_to_survival;
    burnt_vertices = other.burnt_vertices;
    lb = other.lb;
    timestamp = other.timestamp;
    iter = other.iter;
    partial_solution = other.partial_solution;
  }

  Solution &operator=(const Solution &other) {
    if (this == &other) return *this;
    I = other.I;
    fire_arrival_time = other.fire_arrival_time;
    predecessor = other.predecessor;
    fingerprint = other.fingerprint;
    deployment_time = other.deployment_time;
    base_location = other.base_location;
    protected_vertices = other.protected_vertices;
    objv = other.objv;
    time_to_survival = other.time_to_survival;
    burnt_vertices = other.burnt_vertices;
    lb = other.lb;
    timestamp = other.timestamp;
    iter = other.iter;
    partial_solution = other.partial_solution;
    return *this;
  }

  inline TIME get_vertex_delay(VERTEXID v) const { return I.get_delay_value(fingerprint[v]); }

  inline bool has_resource(VERTEXID v) const { return fingerprint[v] != Instance::NULL_DELAY; }

  inline bool has_resource(VERTEXID v, RESOURCE i) const {
    return find(protected_vertices[i].begin(), protected_vertices[i].end(), v) !=
           protected_vertices[i].end();
  }

  inline void set_base_location(RESOURCE i, VERTEXID b) { base_location[i] = b; }

  inline void set_deployment_time(RESOURCE i, TIME t) { deployment_time[i] = t; }

  inline CAPACITY used_capacity(RESOURCE i) const { return protected_vertices[i].size(); }

  RESOURCE get_resource(VERTEXID v) const {
    for (RESOURCE i : I.get_R())
      if (has_resource(v, i)) return i;
    return INF_RESOURCE;
  }

  void allocate_resource(RESOURCE i, VERTEXID v) {
    protected_vertices[i].push_back(v);
    fingerprint[v] = I.get_delay_ID(i);
  }

  void allocate_resource(VERTEXID v) { fingerprint[v] = I.get_delay_ID(0); }

  void deallocate_resource(RESOURCE i, VERTEXID v) {
    protected_vertices[i].remove(v);
    fingerprint[v] = Instance::NULL_DELAY;
  }

  void deallocate_resource(VERTEXID v) { fingerprint[v] = Instance::NULL_DELAY; }

  inline string get_fingerprint() { return fingerprint; }

  void update() { compute_shortest_path_forest(*this); }

  void reset() {
    objv = INF_VALUE;
    lb = 0;
    burnt_vertices = INF_VERTEXID;
    time_to_survival = INF_TIME;
    timestamp = 0;
    iter = 0;
    if (partial_solution) {
      fire_arrival_time = vector<TIME>(I.get_num_vertices(), INF_TIME);
      predecessor = vector<VERTEXID>(I.get_num_vertices(), INF_VERTEXID);
      fingerprint = string(I.get_num_vertices(), Instance::NULL_DELAY);
      deployment_time = vector<TIME>(I.get_num_resources(), INF_TIME);
      base_location = vector<VERTEXID>(I.get_num_resources(), INF_VERTEXID);
      protected_vertices = vector<list<VERTEXID>>(I.get_num_resources());
    } else {
      fill(fire_arrival_time.begin(), fire_arrival_time.end(), INF_TIME);
      fill(predecessor.begin(), predecessor.end(), INF_VERTEXID);
      fill(fingerprint.begin(), fingerprint.end(), Instance::NULL_DELAY);
      fill(deployment_time.begin(), deployment_time.end(), INF_TIME);
      fill(base_location.begin(), base_location.end(), INF_VERTEXID);
      for (auto &l : protected_vertices) l.clear();
      partial_solution = false;
    }
  }

  bool is_trivial_solution() {
    for (RESOURCE i : I.get_R())
      if (protected_vertices[i].size() > 0) return false;
    return true;
  }

  /* -------------------------------------------- PLUMBING FUNCTIONS
   * -------------------------------------------- */
  static void load_solution(const string &file, Solution &S) {
    Instance &I = S.I;
    ifstream f(file);
    if (!f.good()) {
      cerr << "Could not open initial solution file." << endl;
      exit(1);
    }
    json sol = json::parse(f);
    f.close();
    S = Solution(I, true, false);
    S.objv = sol["objv"].get<VALUE>();
    for (auto &[key, value] : sol["allocation"].items()) {
      RESOURCE i = stoi(key);
      if (value["base"].type() == json::value_t::number_unsigned)
        S.set_base_location(i, value["base"]);
      else if (S.I.get_range(i) < INF_DISTANCE) {
        cout << "Resource " << i << " (range " << S.I.get_range(i)
             << ") does not have a base location." << endl;
        exit(1);
      }
      if (value["time"].type() == json::value_t::number_float)
        S.set_deployment_time(i, value["time"]);
      else {
        cout << "Resource " << i << " does not have a deployment time." << endl;
        cout << value["time"] << endl;
        exit(1);
      }
      for (VERTEXID u : value["protected"].get<vector<VERTEXID>>()) {
        S.allocate_resource(i, u);
      }
    }
    compute_shortest_path_forest(S);
    S.partial_solution = false;
    Solution::check_feasibility(S);
  }

  static void write_solution(ostream &fout, Solution &S) {
    Instance &I = S.I;
    json data;
    data["objv"] = S.objv;
    data["allocation"] = json::object();
    for (RESOURCE i : I.get_R()) {
      if (ff(I.get_range(i)) < ff(INF_DISTANCE))
        data["allocation"][to_string(i)] = {{"time", S.deployment_time[i]},
                                            {"base", S.base_location[i]},
                                            {"protected", json::array()}};
      else
        data["allocation"][to_string(i)] = {
            {"time", S.deployment_time[i]}, {"base", "NA"}, {"protected", json::array()}};
      for (VERTEXID v : S.protected_vertices[i])
        data["allocation"][to_string(i)]["protected"].push_back(v);
    }
    string output_str = data.dump(4);
    fout << output_str << std::endl;
  }

  static void print_solution(ostream &fout, Solution &S) {
    Instance &I = S.I;
    auto x_range = I.get_x_range();
    auto y_range = I.get_y_range();
    auto xy_dist = I.get_xy_distance();
    for (int x = x_range.first; x <= x_range.second; x += xy_dist) {
      for (int y = y_range.first; y <= y_range.second; y += xy_dist) {
        VERTEXID v = I.get_vertex_ID(COORDINATE(x, y));
        if (v == INF_VERTEXID)
          fout << ".";
        else if (I.inI(v))
          fout << "#";
        else if (S.has_resource(v))
          fout << "0";
        else if (ff(S.fire_arrival_time[v]) < ff(I.get_optimization_horizon()))
          fout << "x";
        else
          fout << ".";
      }
      fout << endl;
    }
  }

  static bool check_feasibility(Solution &S) {
    Instance &I = S.I;
    TIME H = I.get_optimization_horizon();
    // Save the reported objective value, time to survival, and fire arrival
    // time
    VALUE tmp_obvj = S.objv;
    TIME tmp_time_to_survival = S.time_to_survival;
    vector<TIME> tmp_fire_arrival_time = S.fire_arrival_time;
    // Recompute the shortest path forest
    Solution::compute_shortest_path_forest(S);

    // Check the reported fire arrival time
    for (VERTEXID v : I.get_Vf()) {
      if (ff(tmp_fire_arrival_time[v]) > ff(S.fire_arrival_time[v])) {
        cout << "Reported fire arrival time at vertex " << v
             << " is exceeds correct fire arrival time." << endl;
        cout << "	Reported: " << tmp_fire_arrival_time[v] << endl;
        cout << "	Correct: " << S.fire_arrival_time[v] << endl;
        exit(1);
      }
    }
    // Check the reported objective value
    if (ff(tmp_obvj) != ff(S.objv)) {
      cout << "Reported objective value is incorrect." << endl;
      cout << "	Reported: " << tmp_obvj << endl;
      cout << "	Correct: " << S.objv << endl;
      if (ff(tmp_obvj) < ff(S.objv)) exit(1);
      // The solver may report a higher objective value than the correct one by
      // setting y_v variables to 1 even though the associated vertex is not
      // burnt.
    }
    // Check the reported time to survival
    if (ff(tmp_time_to_survival) != ff(S.time_to_survival)) {
      cout << "Reported time to survival is incorrect." << endl;
      cout << "	Reported: " << tmp_time_to_survival << endl;
      cout << "	Correct: " << S.time_to_survival << endl;
      exit(1);
    }
    // Check if the resource allocation is feasible
    set<VERTEXID> has_resource;
    for (RESOURCE i : I.get_R()) {
      TIME release_time = I.get_release_time(i);
      TIME delay = I.get_resource_delay(i);
      TIME expiration_time = I.get_expiration_time(i);
      TIME safety_time_base_location = I.get_base_safety_time(i);
      TIME safety_time_protected_vertices = I.get_protection_safety_time(i);
      DISTANCE safety_distance_base_location = I.get_base_safety_distance(i);
      DISTANCE safety_distance_protected_vertices = I.get_protection_safety_distance(i);
      CAPACITY capacity = I.get_capacity(i);
      DISTANCE range = I.get_range(i);
      TIME deployment_time = S.deployment_time[i];
      VERTEXID base_location = S.base_location[i];
      CAPACITY num_protected = S.protected_vertices[i].size();
      list<VERTEXID> &protected_vertices = S.protected_vertices[i];

      // Check if the deployment time makes sense
      if (deployment_time == INF_TIME && num_protected > 0) {
        cout << "Resource " << i << " protected " << num_protected
             << " vertices without being deployed." << endl;
        exit(1);
      }
      if (deployment_time == INF_TIME) continue;
      if (ff(deployment_time) < ff(release_time)) {
        cout << "Resource " << i << " was deployed before being released." << endl;
        cout << "	Reported deployment time: " << deployment_time << endl;
        cout << "	Release time: " << release_time << endl;
        exit(1);
      }
      if (ff(deployment_time) > ff(H)) {
        cout << "Resource " << i << " was deployed after time horizon." << endl;
        cout << "	Reported deployment time: " << deployment_time << endl;
        cout << "	Time horizon: " << H << endl;
        exit(1);
      }
      // Check if the capacity of the resource was respected
      if (capacity < num_protected) {
        cout << "The capacity of resource " << i << " was exceeded." << endl;
        cout << "       Capacity: " << capacity << endl;
        cout << "       Number of protected vertices: " << num_protected << endl;
        exit(1);
      }
      // Check if the base location of the resource is feasible
      if (range == INF_DISTANCE) {
        if (base_location != INF_VERTEXID) {
          cout << "Resource " << i << " has a base location without having a maximum range."
               << endl;
          exit(1);
        }
      } else {
        if (base_location == INF_VERTEXID) {
          cout << "Resource " << i << " does not have a base location." << endl;
          exit(1);
        }
        // Check if the base location is in Vb
        if (!I.inVb(base_location, i)) {
          cout << "Vertex " << base_location << " cannot be the base location of resource " << i
               << endl;
          exit(1);
        }
        // Safety constraints
        for (VERTEXID v : I.get_Vf()) {
          if (ff(I.distance(v, base_location)) <= ff(safety_distance_base_location)) {
            if (ff(S.fire_arrival_time[v]) < ff(deployment_time + safety_time_base_location)) {
              cout << "A vertex in the safety radius of the base location of "
                      "resource "
                   << i << " does not respect the safety time." << endl;
              cout << "       Base location: " << base_location << " ("
                   << I.get_vertex_signature(base_location) << ")" << endl;
              cout << "       Safety time: " << safety_time_base_location << endl;
              cout << "       Safety distance: " << safety_distance_base_location << endl;
              cout << "       Problematic vertex: " << v << " (" << I.get_vertex_signature(v) << ")"
                   << endl;
              cout << "       Fire arrival time to the base location: "
                   << S.fire_arrival_time[base_location] << endl;
              cout << "       Fire arrival time to " << v << ": "
                   << S.fire_arrival_time[base_location] << endl;
              cout << "       Distance between the vertices: " << I.distance(v, base_location)
                   << endl;
              exit(1);
            }
          }
        }
      }
      // Check if the protected vertices are feasible
      for (VERTEXID v : protected_vertices) {
        // Check if v can be protected by i
        if (!I.inVp(v, i)) {
          cout << "Resource " << i << " cannot protect vertex " << v << endl;
          exit(1);
        }
        // Check if v already received a resource
        if (has_resource.find(v) != has_resource.end()) {
          cout << "Vertex " << v << " received more than one resource." << endl;
          exit(1);
        } else {
          has_resource.insert(v);
        }
        // Check if v respects the maximum range of the resource
        if (range != INF_DISTANCE && ff(range) < ff(I.distance(v, base_location))) {
          cout << "The maximum range of resource " << i << " was exceeded." << endl;
          cout << "       Distance from protected vertex " << v << " to the base location "
               << base_location << ": " << S.I.distance(v, base_location) << endl;
          cout << "       Range of resource " << i << ": " << range << endl;
          exit(1);
        }
        // Check if the safety time is respected
        for (VERTEXID u : I.get_Vf()) {
          if (ff(I.distance(v, u)) <= ff(safety_distance_protected_vertices)) {
            if (ff(S.fire_arrival_time[u]) < ff(deployment_time + safety_time_protected_vertices)) {
              cout << "A vertex in the safety radius of the protected vertex " << v
                   << " does not respect the safety time." << endl;
              cout << "       Resource: " << i << endl;
              cout << "       Protected vertex: " << v << " (" << I.get_vertex_signature(v) << ")"
                   << endl;
              cout << "       Safety time: " << safety_time_protected_vertices << endl;
              cout << "       Safety distance: " << safety_distance_protected_vertices << endl;
              cout << "       Problematic vertex: " << u << " (" << I.get_vertex_signature(u) << ")"
                   << endl;
              cout << "       Fire arrival time to the protected vertex: " << S.fire_arrival_time[v]
                   << endl;
              cout << "       Fire arrival time to " << u << ": " << S.fire_arrival_time[u] << endl;
              cout << "       Distance between the vertices: " << I.distance(v, u) << endl;
              exit(1);
            }
          }
        }
        // Check if the expiration time is respected
        if (ff(deployment_time + expiration_time) < ff(S.fire_arrival_time[v])) {
          cout << "Resource " << i << " was deployed to vertex " << v << " too early." << endl;
          cout << "       Fire arrival time to " << v << ": " << S.fire_arrival_time[v] << endl;
          cout << "       Resource deployment time: " << deployment_time << endl;
          cout << "       Expiration time: " << expiration_time << endl;
          exit(1);
        }
        // Check if the internal data structures of S assign the correct delay
        // value to vertex v
        if (ff(I.get_delay_value(S.fingerprint[v])) != ff(delay)) {
          cout << "Reported value of Delta for node " << v << " is incorrect." << endl;
          cout << "	Reported: " << I.get_delay_value(S.fingerprint[v]) << endl;
          cout << "	Correct: " << delay << endl;
          exit(1);
        }
      }
    }
    return true;
  }

  static void compute_shortest_path_forest(Solution &S) {
    static MyHeap heap;
    Instance &I = S.I;
    TIME H = I.get_optimization_horizon();
    if (!heap.is_initialized()) heap.initialize(I.get_num_vertices());
    heap.clear();
    if (S.fire_arrival_time.size() < I.get_num_vertices())
      S.fire_arrival_time.resize(I.get_num_vertices());
    if (S.predecessor.size() < I.get_num_vertices()) S.predecessor.resize(I.get_num_vertices());
    fill(S.fire_arrival_time.begin(), S.fire_arrival_time.end(), INF_TIME);
    fill(S.predecessor.begin(), S.predecessor.end(), INF_VERTEXID);
    for (VERTEXID i : I.get_I()) {
      heap.insertElement(i, 0);
      S.fire_arrival_time[i] = 0;
      S.predecessor[i] = i;
    }
    S.burnt_vertices = 0;
    S.objv = 0;
    S.time_to_survival = 0;
    while (!heap.empty()) {
      VERTEXID u = heap.findAndDeleteMinElement();
      TIME a_u = S.fire_arrival_time[u];
      TIME Delta_u = S.get_vertex_delay(u);
      if (ff(a_u) < ff(H)) {
        S.burnt_vertices++;
        S.objv += I.get_value(u);
      }
      S.time_to_survival += Solution::compute_time_to_survival(H, a_u);
      for (auto &[v, t] : I.get_outgoing_arcs(u)) {
        TIME t_uv = t + Delta_u;
        if (ff(S.fire_arrival_time[v]) > ff(a_u + t_uv)) {
          heap.adjustHeap(v, a_u + t_uv);
          S.fire_arrival_time[v] = a_u + t_uv;
          S.predecessor[v] = u;
        }
      }
    }
  }

  static void update_subtree(Solution &S, ranges::input_range auto &&protected_vertices,
                             vector<tuple<VERTEXID, VERTEXID, TIME>> &affected_vertices) {
    /*
        Assumptions:
            Increment is always a positive constant
            All propagation times are positive numbers
    */
    static MyHeap heap;
    vector<VERTEXID> Q;
    Instance &I = S.I;
    if (!heap.is_initialized()) heap.initialize(I.get_num_vertices());
    heap.clear();
    affected_vertices.clear();
    // Identify vertices directly affected by the modification
    for (VERTEXID u : protected_vertices)
      for (auto &[v, t_uv] : I.get_outgoing_arcs(u))
        if (S.predecessor[v] == u) heap.insertElement(v, S.fire_arrival_time[v]);
    // Identify all affected vertices in order of fire arrival time. Set their
    // fire arrival times to infinity
    while (!heap.empty()) {
      VERTEXID u = heap.findAndDeleteMinElement();
      VERTEXID pred_u = S.predecessor[u];
      for (auto &[v, t] : I.get_incoming_arcs(u)) {
        TIME t_vu = t + S.get_vertex_delay(v);
        if (ff(S.fire_arrival_time[v]) < ff(INF_TIME) &&
            ff(S.fire_arrival_time[u]) == ff(S.fire_arrival_time[v] + t_vu)) {
          affected_vertices.emplace_back(u, S.predecessor[u], S.fire_arrival_time[u]);
          S.predecessor[u] = v;
          break;
        }
      }
      if (pred_u == S.predecessor[u]) {
        affected_vertices.emplace_back(u, S.predecessor[u], S.fire_arrival_time[u]);
        Q.push_back(u);
        S.fire_arrival_time[u] = INF_TIME;
        for (auto &[v, t_uv] : I.get_outgoing_arcs(u)) {
          if (S.predecessor[v] == u && ff(S.fire_arrival_time[v]) != ff(INF_TIME))
            heap.adjustHeap(v, S.fire_arrival_time[v]);
        }
      }
    }
    // Update fire arrival time of vertices that are in the border between the
    // subtree of affected vertices and the rest of the shortest-path tree
    for (VERTEXID u : Q) {
      for (auto &[v, t] : I.get_incoming_arcs(u)) {
        TIME t_vu = t + S.get_vertex_delay(v);
        if (ff(S.fire_arrival_time[v]) < ff(INF_TIME) &&
            ff(S.fire_arrival_time[u]) > ff(S.fire_arrival_time[v] + t_vu)) {
          S.fire_arrival_time[u] = S.fire_arrival_time[v] + t_vu;
          S.predecessor[u] = v;
        }
      }
      if (ff(S.fire_arrival_time[u]) != ff(INF_TIME)) heap.adjustHeap(u, S.fire_arrival_time[u]);
    }
    // Update fire arrival time of the remaining vertices
    while (!heap.empty()) {
      VERTEXID u = heap.findAndDeleteMinElement();
      for (auto &[v, t] : I.get_outgoing_arcs(u)) {
        TIME t_uv = t + S.get_vertex_delay(u);
        if (ff(S.fire_arrival_time[v]) > ff(S.fire_arrival_time[u] + t_uv)) {
          S.fire_arrival_time[v] = S.fire_arrival_time[u] + t_uv;
          S.predecessor[v] = u;
          heap.adjustHeap(v, S.fire_arrival_time[v]);
        }
      }
    }
  }

  static void allocate_resources(Solution &S, RESOURCE i, ranges::input_range auto &&C) {
    Instance &I = S.I;
    vector<tuple<VERTEXID, VERTEXID, TIME>> affected_vertices;
    TIME H = I.get_optimization_horizon();
    // Update S with new resources
    for (VERTEXID u : C) {
      S.allocate_resource(i, u);
    }
    S.set_deployment_time(i, S.I.get_release_time(i));
    S.set_base_location(i, INF_VERTEXID);
    // Update subtree
    Solution::update_subtree(S, C, affected_vertices);
    // Update 'objv' and 'burnt_vertices'
    for (auto &[u, pred_u, a_u] : affected_vertices) {
      if (ff(S.fire_arrival_time[u]) >= ff(H) && ff(a_u) < ff(H)) {
        S.objv -= I.get_value(u);
        S.burnt_vertices--;
      }
      S.time_to_survival -= Solution::compute_time_to_survival(H, a_u);
      S.time_to_survival += Solution::compute_time_to_survival(H, S.fire_arrival_time[u]);
    }
  }

  static void compute_heuristic_values(Solution &S, RESOURCE i, vector<VERTEXID> &C, VALUE &objv,
                                       TIME &time_to_survival) {
    Instance &I = S.I;
    TIME H = I.get_optimization_horizon();
    vector<tuple<VERTEXID, VERTEXID, TIME>> affected_vertices;
    VALUE prev_objv = S.objv;
    TIME prev_time_to_survival = S.time_to_survival;

    objv = S.objv;
    time_to_survival = S.time_to_survival;
    // Update solution
    for (VERTEXID u : C) S.allocate_resource(i, u);
    Solution::update_subtree(S, C, affected_vertices);
    // Compute heuristic values
    for (auto &[u, pred_u, a_u] : affected_vertices) {
      if (ff(S.fire_arrival_time[u]) >= ff(H) && ff(a_u) < ff(H)) objv -= I.get_value(u);
      time_to_survival -= Solution::compute_time_to_survival(H, a_u);
      time_to_survival += Solution::compute_time_to_survival(H, S.fire_arrival_time[u]);
    }
    // Reset solution
    for (VERTEXID u : C) S.deallocate_resource(i, u);
    for (auto &[u, pred_u, a_u] : affected_vertices) {
      S.fire_arrival_time[u] = a_u;
      S.predecessor[u] = pred_u;
    }
    S.objv = prev_objv;
    S.time_to_survival = prev_time_to_survival;
  }

  static TIME compute_time_to_survival(TIME H, TIME a_v) {
    assert(a_v < INF_TIME);
    return max(TIME(0), H - a_v);
  }

  static unsigned hamming_distance(const Solution &A, const Solution &B) {
    Instance &I = A.I;
    unsigned distance = 0;
    for (VERTEXID v : I.get_V())
      if (A.fingerprint[v] != B.fingerprint[v]) distance++;
    return distance;
  }

  static bool compute_greedy_schedule(Solution &S, vector<VERTEXID> P) {
    if (!S.I.is_basic_problem()) {
      cout << "Greedy schedule is only available for basic problems." << endl;
      exit(1);
    }
    Instance &I = S.I;
    vector<RESOURCE> R;
    vector<CAPACITY> C;
    for (VERTEXID v : P) {
      assert(!S.has_resource(v));
      S.allocate_resource(v);
    }
    S.update();
    for (RESOURCE i : I.get_R()) {
      R.push_back(i);
      C.push_back(I.get_capacity(i));
    }
    sort(R.begin(), R.end(),
         [&](RESOURCE i, RESOURCE j) { return I.get_release_time(i) > I.get_release_time(j); });
    sort(P.begin(), P.end(), [&](VERTEXID u, VERTEXID v) {
      return (S.fire_arrival_time[u] > S.fire_arrival_time[v]) ||
             (S.fire_arrival_time[u] == S.fire_arrival_time[v] && u > v);
    });
    int P_idx = P.size() - 1;
    int R_idx = R.size() - 1;
    bool feasible = true;
    while (P_idx >= 0) {
      VERTEXID v = P[P_idx];
      RESOURCE i = R[R_idx];
      if (ff(S.fire_arrival_time[v]) < ff(I.get_release_time(i))) feasible = false;
      S.deallocate_resource(v);
      S.allocate_resource(i, v);
      S.set_deployment_time(i, I.get_release_time(i));
      C[i]--;
      P_idx--;
      if (C[i] == 0) R_idx--;
    }
    return feasible;
  }
};