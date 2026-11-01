
#pragma once

#include <numeric>
#include <string>
using namespace std;

#include <boost/multi_array.hpp>
using namespace boost;

#include "clock.hpp"
#include "common.hpp"
#include "gurobi_c++.h"
#include "instance.hpp"
#include "solution.hpp"

class Model {
 public:
  Model() {}

  virtual ~Model() = default;

  virtual void get_solution(Solution& B) = 0;           // Get the best solution found by the solver
  virtual void load_initial_solution(Solution& S) = 0;  // Load an initial solution from a JSON file
  virtual void build() = 0;                             // Build the model
  virtual void solve() = 0;                             // Solve the model
  virtual void write() = 0;                             // Write the model to a .lp file
  virtual bool is_relaxation() = 0;  // Check if the solved model is a relaxation of the
                                     // original problem (i.e. flag mipRelaxation = true)
  // Statistics and information retrieval methods
  virtual string get_objv() = 0;
  virtual string get_lower_bound() = 0;
  virtual string get_elapsed_time() = 0;
  virtual string get_mip_gap() = 0;
  virtual string get_node_count() = 0;
  virtual string get_solution_count() = 0;
  virtual string get_open_node_count() = 0;
  virtual string get_num_constr() = 0;
  virtual string get_num_vars() = 0;
  virtual string get_num_bin_vars() = 0;
  virtual string get_num_nzs() = 0;
  virtual string get_max_coeff() = 0;
  virtual string get_min_coeff() = 0;
  virtual string get_max_bound() = 0;
  virtual string get_min_bound() = 0;
  virtual string get_max_obj_coeff() = 0;
  virtual string get_min_obj_coeff() = 0;
  virtual string get_max_rhs() = 0;
  virtual string get_min_rhs() = 0;
  virtual string presolve_get_num_constr() = 0;
  virtual string presolve_get_num_vars() = 0;
  virtual string presolve_get_num_bin_vars() = 0;
  virtual string presolve_get_num_nzs() = 0;
  virtual string presolve_get_max_coeff() = 0;
  virtual string presolve_get_min_coeff() = 0;
  virtual string presolve_get_max_bound() = 0;
  virtual string presolve_get_min_bound() = 0;
  virtual string presolve_get_max_obj_coeff() = 0;
  virtual string presolve_get_min_obj_coeff() = 0;
  virtual string presolve_get_max_rhs() = 0;
  virtual string presolve_get_min_rhs() = 0;
  virtual vector<tuple<unsigned, double, double>> get_trajectory() = 0;
};
