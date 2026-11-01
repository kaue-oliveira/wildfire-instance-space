#pragma once

#include <cmath>
using namespace std;

#include "algorithm.hpp"
#include "basic_model.hpp"
#include "common.hpp"
#include "general_model.hpp"
#include "parameters.hpp"
#include "solution.hpp"

class MIP : public Algorithm {
 private:
  Instance &I;
  Solution B;
  Model *model;
  GeneralParameters generalParameters;
  MipParameters mipParameters;

 public:
  MIP(Instance &_I, GeneralParameters _generalParameters, MipParameters _mipParameters)
      : I(_I),
        B(_I, true, false),
        model(nullptr),
        generalParameters(_generalParameters),
        mipParameters(_mipParameters) {
    if (mipParameters.mipModel == "basic")
      model = new BasicModel(I, generalParameters, mipParameters);
    else if (mipParameters.mipModel == "general")
      model = new GeneralModel(I, generalParameters, mipParameters);
    model->build();
    if (mipParameters.mipSaveModel) model->write();
  }

  ~MIP() { delete model; }

  void run() {
    model->solve();
    if (!model->is_relaxation() && stoi(model->get_solution_count()) > 0) {
      model->get_solution(B);
      B.update();
      Solution::check_feasibility(B);
    }
  }

  void get_solution(Solution &S) { S = B; }

  void load_initial_solution(const string &file) {
    Solution S(I);
    Solution::load_solution(file, S);
    Solution::check_feasibility(S);
    model->load_initial_solution(S);
  }

  void write_statistics(ostream &out) {
    if (generalParameters.tuning) {
      cout << B.objv << endl;
      return;
    }
    string traj = "\"[";
    for (const auto &[time, objective, bound] : model->get_trajectory())
      traj = traj + "(" + to_string(time) + "," + to_string(round(objective)) + "," +
             to_string(round(bound)) + "), ";
    if (traj.size() > 2) {
      traj.pop_back();
      traj.pop_back();
    }
    traj = traj + "]\"";
    out << I.get_instance_id() << "," << generalParameters.seed << "," << model->get_objv() << ","
        << model->get_elapsed_time() << "," << B.time_to_survival << ","
        << (B.burnt_vertices == INF_VERTEXID ? "-" : to_string(B.burnt_vertices)) << ","
        << model->get_lower_bound() << "," << model->get_mip_gap() << "," << model->get_node_count()
        << "," << model->get_solution_count() << "," << model->get_open_node_count() << ","
        << model->get_num_constr() << "," << model->get_num_vars() << ","
        << model->get_num_bin_vars() << "," << model->get_num_nzs() << "," << model->get_max_coeff()
        << "," << model->get_min_coeff() << "," << model->get_max_bound() << ","
        << model->get_min_bound() << "," << model->get_max_obj_coeff() << ","
        << model->get_min_obj_coeff() << "," << model->get_max_rhs() << "," << model->get_min_rhs()
        << "," << model->presolve_get_num_constr() << "," << model->presolve_get_num_vars() << ","
        << model->presolve_get_num_bin_vars() << "," << model->presolve_get_num_nzs() << ","
        << model->presolve_get_max_coeff() << "," << model->presolve_get_min_coeff() << ","
        << model->presolve_get_max_bound() << "," << model->presolve_get_min_bound() << ","
        << model->presolve_get_max_obj_coeff() << "," << model->presolve_get_min_obj_coeff() << ","
        << model->presolve_get_max_rhs() << "," << model->presolve_get_min_rhs();
    if (model->is_relaxation())
      out << endl;
    else
      out << "," << traj << endl;
  }

  void write_solution(ostream &out) {
    if (model->is_relaxation())
      cout << "Impossible to save a relaxed solution." << endl;
    else if (model->get_solution_count() == "0")
      cout << "Could not find a feasible solution." << endl;
    else
      Solution::write_solution(out, B);
  }

  void print_solution(ostream &out) {
    if (model->is_relaxation())
      cout << "Impossible to print a relaxed solution." << endl;
    else if (model->get_solution_count() == "0")
      cout << "Could not find a feasible solution." << endl;
    else
      Solution::print_solution(out, B);
  }
};
