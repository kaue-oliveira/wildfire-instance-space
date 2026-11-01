#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
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
#include "cut_model.hpp"
#include "instance.hpp"
#include "parameters.hpp"
#include "solution.hpp"

class CutHeuristic : public Algorithm {
 private:
  Clock clock;
  unsigned total_iterations;        // Number of iterations of the algorithm
  unsigned elapsed_time;            // Clock time at which the algorithm stopped
  unsigned feasible_iterations;     // Number of iterations in which a feasible
                                    // solution was found
  unsigned iterations_to_best;      // Number of iterations to find the best solution
  unsigned time_to_best;            // Clock time at which the best solution was found
  TIME best_time;                   // Binary search time at which the best solution was found
  TIME earliest_feasible_solution;  // Earliest binary search time at which a
                                    // feasible solution was found
  vector<tuple<unsigned, unsigned, VALUE>> trajectory;

  // Parameters
  mt19937 gen;
  Instance &I;
  GeneralParameters generalParameters;
  CutHeuristicParameters cuthParameters;
  Solution A0;
  Solution B;

 public:
  CutHeuristic(Instance &I, GeneralParameters general, CutHeuristicParameters cuth)
      : clock(Clock(false)),
        total_iterations(0),
        elapsed_time(0),
        feasible_iterations(0),
        iterations_to_best(0),
        time_to_best(0),
        best_time(INF_TIME),
        earliest_feasible_solution(INF_TIME),
        I(I),
        generalParameters(general),
        cuthParameters(cuth),
        A0(I, true, true),
        B(I) {
    gen = mt19937(general.seed);
  }

  void find_cut(Solution &S, TIME t_start, vector<VERTEXID> &P) {
    assert(Solution::check_feasibility(S));
    auto available_time =
        min(cuthParameters.cuthSolverTime, generalParameters.timelimit - clock.now());
    vector<RESOURCE> allocation(I.get_num_vertices(), INF_RESOURCE);
    CAPACITY num_resources = 0;
    for (RESOURCE i : I.get_R()) num_resources += I.get_capacity(i);
    CutModel model(S, t_start, num_resources, available_time);
    P.clear();
    model.optimize(P);
  }

  bool compute_greedy_schedule(Solution &S, vector<VERTEXID> &P) {
    bool feasible = Solution::compute_greedy_schedule(S, P);
    return feasible;
  }

  void run() {
    clock.start_clock();
    Solution S = A0;
    TIME t_lb = INF_TIME;
    TIME t_ub = 0;
    vector<VERTEXID> P;
    // Initialize lower and upper bounds for binary search
    for (RESOURCE i : I.get_R()) {
      t_lb = min(t_lb, I.get_release_time(i));
      t_ub = max(t_ub, I.get_release_time(i));
    }
    do {
      total_iterations++;
      S = A0;
      TIME t_mid = (t_lb + t_ub) / 2;
      find_cut(S, t_mid, P);
      bool feasible = compute_greedy_schedule(S, P);
      if (feasible) {
        // Update upper bound
        t_ub = t_mid;
        // Update the best solution found so far
        if (ff(S.objv) < ff(B.objv)) {
          B = S;
          trajectory.push_back({clock.now(), total_iterations, B.objv});
          iterations_to_best = total_iterations;
          time_to_best = clock.now();
          best_time = t_mid;
        }
        // Update statistics
        earliest_feasible_solution = min(earliest_feasible_solution, t_mid);
      } else {
        // Update lower bound
        t_lb = t_mid;
      }
    } while (abs(t_lb - t_ub) > cuthParameters.cuthEpsilon &&
             clock.now() < generalParameters.timelimit);
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
        << "," << total_iterations << "," << feasible_iterations << "," << iterations_to_best << ","
        << time_to_best << "," << best_time << "," << earliest_feasible_solution << ","
        << elapsed_time << "," << traj << endl;
  }

  void write_solution(ostream &out) { Solution::write_solution(out, B); }

  void print_solution(ostream &out) { Solution::print_solution(out, B); }
};
