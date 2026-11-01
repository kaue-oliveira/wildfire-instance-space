#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>
using namespace std;

#include "algorithm.hpp"
#include "clock.hpp"
#include "common.hpp"
#include "instance.hpp"
#include "parameters.hpp"
#include "solution.hpp"

class BeamSearch : public Algorithm {
 private:
  Clock clock;
  ull budget;
  unsigned z;
  unsigned total_iterations;
  unsigned elapsed_time;
  vector<tuple<unsigned, unsigned, VALUE>> trajectory;

  // Parameters
  mt19937 gen;                          // Random number generator
  Instance &I;                          // Instance reference
  GeneralParameters generalParameters;  // General parameters
  BeamSearchParameters ibsParameters;   // Beam search parameters
  TIME free_burning_time;               // The free burning time of the instance
  TIME that;                            // Transition instant

  vector<TIME> T;                 // Sequence of release times
  map<TIME, vector<RESOURCE>> R;  // Map from release time to resources available at that time
  vector<vector<VERTEXID>> extended_neighborhoods;  // Extended neighborhoods of vertices
  Solution A0;                                      // Empty allocation
  Solution B;                                       // Best solution found so far

  /* ----------------- AUXILIARY FUNCTIONS ---------------------------- */
  TIME alpha(TIME t) {
    if (ff(t) >= ff(T.back()))
      return I.get_optimization_horizon();
    else
      return *upper_bound(T.begin(), T.end(), t);
  }

  vector<VERTEXID> get_extended_neighborhood(VERTEXID v) {
    /*
        WARNING: This function assumes that the graph is a square grid
       graph.
    */
    DISTANCE xy_dist = I.get_xy_distance();
    pair<int, int> x_range = I.get_x_range();
    pair<int, int> y_range = I.get_y_range();
    int x_min = x_range.first;
    int x_max = x_range.second;
    int y_min = y_range.first;
    int y_max = y_range.second;
    vector<VERTEXID> neighborhood;
    COORDINATE_3D v_coord = I.get_coordinate(v);
    vector<pair<int, int>> neighbors = {
        {xy_dist, 0},       {0, xy_dist},        {-xy_dist, 0},       {0, -xy_dist},
        {xy_dist, xy_dist}, {xy_dist, -xy_dist}, {-xy_dist, xy_dist}, {-xy_dist, -xy_dist}};
    for (auto [dx, dy] : neighbors) {
      COORDINATE_3D u = COORDINATE_3D(v_coord.x + dx, v_coord.y + dy, v_coord.z);
      if ((x_min <= u.x) && (u.x <= x_max) && (y_min <= u.y) && (u.y <= y_max))
        if (I.coordinate_exists(u)) neighborhood.push_back(I.get_vertex_ID(u));
    }
    return neighborhood;
  }

  /* ----------------- BEAM SEARCH ---------------------------- */
  void step(Solution &S, RESOURCE i, vector<Solution> &E) {
    set<VERTEXID> F;
    set<VERTEXID> N;
    set<VERTEXID> F_copy;
    set<VERTEXID> N_copy;
    vector<VERTEXID> selection;
    vector<vector<VERTEXID>> candidates;
    vector<pair<double, unsigned>> scores;
    set<string> solution_set;
    uniform_int_distribution<> dist(0, 0);
    uniform_real_distribution<double> dist01(0, 1);
    TIME H = I.get_optimization_horizon();
    TIME release_time = I.get_release_time(i);
    DELAYID delay_id = I.get_delay_ID(i);
    CAPACITY capacity = I.get_capacity(i);
    string S_fingerprint = S.get_fingerprint();
    bool last_batch = alpha(release_time) >= H;

    // 1. Compute f(t, z)
    TIME fp_cutoff = release_time;
    TIME lb = release_time, ub = release_time;
    unsigned zp = ceil((z + 1) / 2.0);
    while (zp >= 1) {
      lb = alpha(lb);
      zp--;
    }
    zp = ceil((z + 2) / 2.0);
    while (zp >= 1) {
      ub = alpha(ub);
      zp--;
    }
    if (ibsParameters.ibsFirePerimenterThreshold)  // If ibsFirePerimenterThreshold
                                                   // is true, we ignore ignore the
                                                   // threshold
      fp_cutoff = 0.5 * lb + 0.5 * ub;
    else
      fp_cutoff = H;
    // 2. Build F and N
    for (VERTEXID u : I.get_Vf()) {
      if (!S.has_resource(u) && ff(S.fire_arrival_time[u]) >= ff(release_time) &&
          ff(S.fire_arrival_time[u]) <= ff(fp_cutoff)) {
        F.insert(u);
        F_copy.insert(u);
      }
    }
    for (VERTEXID u : I.get_Vf())
      if (S.has_resource(u))
        for (VERTEXID v : extended_neighborhoods[u])  // TODO: double check if
                                                      // this is right
          if (F.contains(v)) {
            N_copy.insert(v);
            N.insert(v);
          }
    // 3. Initialize solution set
    for (Solution &s : E) solution_set.insert(s.get_fingerprint());
    // 4. Step
    unsigned trials =
        ibsParameters.ibsManyTrials ? ibsParameters.ibsC * F.size() : ibsParameters.ibsEta;
    for (unsigned trial = 0; trial < trials; trial++) {
      // 4.1. Expand
      for (unsigned res = 0; res < capacity; res++) {
        VERTEXID u = INF_VERTEXID;
        if (!N.empty() && dist01(gen) < ibsParameters.ibsP) {
          dist.param(uniform_int_distribution<>::param_type(0, N.size() - 1));
          u = *next(N.begin(), dist(gen));
        } else if (!F.empty()) {
          dist.param(uniform_int_distribution<>::param_type(0, F.size() - 1));
          u = *next(F.begin(), dist(gen));
        }
        if (u == INF_VERTEXID) break;
        selection.push_back(u);
        F.erase(u);
        N.erase(u);
        for (VERTEXID v : extended_neighborhoods[u])
          if (F.contains(v)) N.insert(v);
      }
      assert(selection.size() <= capacity);
      // 4.2. Update the fingerprint of S and check whether the new
      // solution is in the solution set
      for (VERTEXID v : selection) S_fingerprint[v] = delay_id;
      if (!solution_set.contains(S_fingerprint)) {
        solution_set.insert(S_fingerprint);
        double h1_objv, h2_tts;
        Solution::compute_heuristic_values(S, i, selection, h1_objv, h2_tts);
        budget++;
        candidates.push_back(selection);
        if (release_time >= that || last_batch)
          scores.emplace_back(h1_objv, candidates.size() - 1);
        else
          scores.emplace_back(h2_tts, candidates.size() - 1);
      }
      // 4.3. Reset the fingerprint
      for (VERTEXID v : selection) S_fingerprint[v] = Instance::NULL_DELAY;
      // 4.4. Reset N, F, and selection
      N = N_copy;
      F = F_copy;
      selection.clear();
      if (clock.now() >= generalParameters.timelimit) break;
    }
    // 5. Sort all the expansions
    sort(scores.begin(), scores.end(),
         [](pair<double, unsigned> &a, pair<double, unsigned> &b) { return a.first < b.first; });

    // 6. Add the best solutions to E
    if (candidates.empty()) {
      E.emplace_back(S);
    } else {
      for (unsigned idx = 0; idx < min(candidates.size(), size_t(ibsParameters.ibsEta)); idx++) {
        E.emplace_back(S);
        Solution::allocate_resources(E.back(), i, candidates[scores[idx].second]);
        budget++;
      }
    }
  }

  void prune(vector<Solution> &A, vector<Solution> &E, RESOURCE i) {
    bool last_batch = alpha(I.get_release_time(i)) >= I.get_optimization_horizon();
    // Sort E
    if (I.get_release_time(i) >= that || last_batch)
      sort(E.begin(), E.end(), [](Solution &a, Solution &b) { return a.objv < b.objv; });
    else
      sort(E.begin(), E.end(),
           [](Solution &a, Solution &b) { return a.time_to_survival < b.time_to_survival; });
    // Add best solutions to A
    A.clear();
    for (unsigned i = 0; i < min(size_t(ibsParameters.ibsBeta), E.size()); i++)
      A.emplace_back(E[i]);
  }

 public:
  BeamSearch(Instance &I, GeneralParameters general, BeamSearchParameters ibs)
      : clock(Clock(false)),
        budget(0),
        z(0),
        total_iterations(0),
        elapsed_time(0),
        I(I),
        generalParameters(general),
        ibsParameters(ibs),
        A0(I, true, true),
        B(I) {
    gen = mt19937(general.seed);
    // Compute the free burning time
    auto filtered_view =
        A0.fire_arrival_time | std::views::filter([](TIME time) { return time != INF_TIME; });
    auto max_it = std::ranges::max_element(filtered_view);
    if (max_it != std::ranges::end(filtered_view))
      free_burning_time = *max_it;
    else
      free_burning_time = INF_TIME;
    // Compute transition instant
    that = ibsParameters.ibsPhat * free_burning_time;
    // Build T and R
    set<TIME> T_tmp;
    for (RESOURCE i : I.get_R()) {
      TIME t = I.get_release_time(i);
      if (R.find(t) == R.end())
        R[t] = {i};
      else
        R[t].push_back(i);
      T_tmp.insert(t);
    }
    T = vector<TIME>(T_tmp.begin(), T_tmp.end());
    // Initialize extended neighborhoods
    for (VERTEXID v : I.get_Vf()) extended_neighborhoods.push_back(get_extended_neighborhood(v));
  }

  void run() {
    clock.start_clock();
    while (clock.now() < generalParameters.timelimit &&
           total_iterations < generalParameters.max_iterations &&
           budget < ibsParameters.ibsMaxBudget && B.objv > generalParameters.target_objv) {
      // Build search tree
      vector<Solution> A = {A0};
      for (TIME t : T) {
        for (RESOURCE i : R[t])  // In practice, each t is associated with a single
                                 // resource i, so this for loop will run only once
        {
          vector<Solution> E;
          for (Solution &s : A) {
            step(s, i, E);
            if (clock.now() >= generalParameters.timelimit) break;
          }
          prune(A, E, i);
        }
        if (clock.now() >= generalParameters.timelimit) break;
      }
      total_iterations++;
      // Update best solution
      trajectory.push_back({clock.now(), total_iterations, A[0].objv});
      if (A[0].objv < B.objv) {
        B = A[0];
        B.timestamp = total_iterations;
        B.burnt_vertices = B.fire_arrival_time.size();
      } else
        z = (z + 1) % ibsParameters.ibsZmax;
      // Update trajectory
      if (generalParameters.verbose)
        cout << std::setprecision(10) << clock.now() << " " << A[0].objv << " " << budget << endl;
    }
    elapsed_time = clock.now();
    Solution::check_feasibility(B);
  }

  void get_solution(Solution &S) { S = B; }

  void load_initial_solution(const string &file) {
    Solution::load_solution(file, B);
    Solution::check_feasibility(B);
  }

  void write_statistics(ostream &out) {
    string traj = "\"[";
    if (generalParameters.tuning) {
      cout << B.objv << endl;
      return;
    }
    for (const auto &[time, iteration, objv] : trajectory)
      traj =
          traj + "(" + to_string(time) + "," + to_string(iteration) + "," + to_string(objv) + "), ";
    if (traj.size() > 2) {
      traj.pop_back();
      traj.pop_back();
    }
    traj = traj + "]\"";
    out << std::setprecision(10) << I.get_instance_id() << "," << generalParameters.seed << ","
        << ibsParameters.ibsBeta << "," << ibsParameters.ibsEta << "," << ibsParameters.ibsP << ","
        << ibsParameters.ibsPhat << "," << ibsParameters.ibsC << "," << ibsParameters.ibsZmax << ","
        << B.objv << "," << B.timestamp << "," << B.time_to_survival << "," << B.burnt_vertices
        << "," << total_iterations << "," << elapsed_time << "," << budget << "," << traj << endl;
  }

  void write_solution(ostream &out) { Solution::write_solution(out, B); }

  void print_solution(ostream &out) { Solution::print_solution(out, B); }
};