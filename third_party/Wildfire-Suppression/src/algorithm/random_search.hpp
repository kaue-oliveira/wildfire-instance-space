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

class RandomSearch : public Algorithm {
 private:
  Clock clock;
  unsigned total_iterations;
  unsigned elapsed_time;
  vector<tuple<unsigned, unsigned, VALUE>> trajectory;

  // Parameters
  mt19937 gen;
  Instance &I;
  GeneralParameters generalParameters;
  vector<TIME> T;                 // Sequence of release times
  map<TIME, vector<RESOURCE>> R;  // Map from release time to resources available at that time
  Solution A0;                    // Empty allocation
  Solution B;                     // Best solution found so far

 public:
  RandomSearch(Instance &I, GeneralParameters general)
      : clock(Clock(false)),
        total_iterations(0),
        elapsed_time(0),
        I(I),
        generalParameters(general),
        A0(I, true, true),
        B(I) {
    gen = mt19937(general.seed);
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
  }

  void run() {
    clock.start_clock();
    Solution S = A0;
    while (clock.now() < generalParameters.timelimit &&
           total_iterations < generalParameters.max_iterations &&
           B.objv > generalParameters.target_objv) {
      for (TIME t : T) {
        for (RESOURCE i : R[t]) {
          vector<VERTEXID> C;
          for (VERTEXID v : I.get_Vf())
            if (!S.has_resource(v) && S.fire_arrival_time[v] >= t) C.push_back(v);
          shuffle(C.begin(), C.end(), gen);
          Solution::allocate_resources(
              S, i, std::views::take(C, min(size_t(I.get_capacity(i)), C.size())));
        }
        if (clock.now() >= generalParameters.timelimit) break;
      }
      total_iterations++;
      if (S.objv < B.objv) {
        B = S;
        B.timestamp = total_iterations;
        B.burnt_vertices = B.fire_arrival_time.size();
        trajectory.push_back({clock.now(), total_iterations, S.objv});
        if (generalParameters.verbose)
          cout << std::setprecision(10) << clock.now() << " " << S.objv << " " << endl;
      }
      S = A0;
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
        << B.objv << "," << B.timestamp << "," << B.time_to_survival << "," << B.burnt_vertices
        << "," << total_iterations << "," << elapsed_time << "," << traj << endl;
  }

  void write_solution(ostream &out) { Solution::write_solution(out, B); }

  void print_solution(ostream &out) { Solution::print_solution(out, B); }
};