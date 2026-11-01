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
#include "model.hpp"
#include "solution.hpp"

class BasicModel : public Model {
 protected:
  class Trajectory : public GRBCallback {
   public:
    vector<tuple<unsigned, double, double>> trajectory;
    Clock clock;
    Trajectory() : clock(Clock(false)) {}
    void init() { clock.start_clock(); }

   protected:
    void callback() {
      try {
        if (where == GRB_CB_MIPSOL) {
          if (getDoubleInfo(GRB_CB_MIPSOL_OBJBST) < 1e10 && getDoubleInfo(GRB_CB_MIPSOL_OBJBND) > 0)
            trajectory.push_back({clock.now(), getDoubleInfo(GRB_CB_MIPSOL_OBJBST),
                                  getDoubleInfo(GRB_CB_MIPSOL_OBJBND)});
        }
      } catch (GRBException e) {
        cout << "Error number: " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
      } catch (...) {
        cout << "Error during callback" << endl;
      }
    }
  };

  // Parameters
  Instance &I;
  GeneralParameters generalParameters;
  MipParameters mipParameters;
  Trajectory trajectory;
  // Model
  GRBEnv *env;
  GRBModel *model;
  GRBModel *presolve;
  GRBModel *relaxed_model;
  GRBModel *model_handle;
  map<VERTEXID, GRBVar> a;
  map<pair<RESOURCE, VERTEXID>, GRBVar> r;
  map<VERTEXID, GRBVar> y;
  // Stats
  bool solved;

  void add_variables_a() {
    for (VERTEXID v : I.get_V())
      a[v] = model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "a_" + to_string(v));
  }

  void add_variables_y() {
    for (VERTEXID v : I.get_V())
      y[v] = model->addVar(0.0, 1.0, 1.0, GRB_BINARY, "y_" + to_string(v));
  }

  void add_variables_r() {
    for (RESOURCE i : I.get_R())
      for (VERTEXID v : I.get_V())
        r[{i, v}] =
            model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "r_" + to_string(i) + "_" + to_string(v));
  }

  // 1 - Fire propagation
  void add_constraints_C1() {
    for (VERTEXID v : I.get_Vf()) {
      if (I.inI(v))
        model->addConstr(a[v] == 0, "C1.a." + to_string(v));
      else {
        for (auto &[u, t_uv] : I.get_incoming_arcs(v)) {
          GRBLinExpr expr = 0;
          for (RESOURCE i : I.get_R()) {
            TIME delay = I.get_resource_delay(i);
            expr += delay * r[{i, u}];
          }
          model->addConstr(a[v] <= a[u] + t_uv + expr, "C1.b" + to_string(v) + "_" + to_string(u));
        }
      }
    }
  }

  // 2 -  Each resource cannot protect more vertices than its capacity
  void add_constraints_C2() {
    for (RESOURCE i : I.get_R()) {
      CAPACITY capacity = I.get_capacity(i);
      GRBLinExpr sum_r_i_v = 0;
      for (VERTEXID v : I.get_V()) sum_r_i_v += r[{i, v}];
      model->addConstr(sum_r_i_v <= capacity, "C2." + to_string(i));
    }
  }

  // 3 - Each flammable vertex can receive at most one resource
  void add_constraints_C3() {
    for (VERTEXID v : I.get_V()) {
      GRBLinExpr sum_r_i_v = 0;
      for (RESOURCE i : I.get_R()) sum_r_i_v += r[{i, v}];
      model->addConstr(sum_r_i_v <= 1, "C3." + to_string(v));
    }
  }

  // 4 - A burned vertex cannot receive a resource
  void add_constraints_C4() {
    for (RESOURCE i : I.get_R()) {
      TIME release_time = I.get_release_time(i);
      for (VERTEXID v : I.get_V()) {
        model->addConstr(a[v] >= r[{i, v}] * release_time,
                         "C5." + to_string(i) + "_" + to_string(v));
      }
    }
  }

  // 5 - Burned vertices
  void add_constraints_C5() {
    TIME H = I.get_optimization_horizon();
    for (VERTEXID v : I.get_V()) {
      model->addConstr(y[v] >= 1 - a[v] / H, "C6." + to_string(v));
    }
  }

 public:
  BasicModel(Instance &_I, GeneralParameters gp, MipParameters mp)
      : I(_I), generalParameters(gp), mipParameters(mp) {
    try {
      env = new GRBEnv(true);
      env->set(GRB_DoubleParam_MemLimit, generalParameters.memlimit);
      env->start();
      model = new GRBModel(env);
      presolve = nullptr;
      relaxed_model = nullptr;
      model->set(GRB_IntParam_OutputFlag, generalParameters.verbose);
      model->set(GRB_DoubleParam_TimeLimit, generalParameters.timelimit);
      model->set(GRB_IntParam_Seed, generalParameters.seed);
      model->set(GRB_IntParam_Threads, 1);
      model->set(GRB_IntParam_Symmetry, mipParameters.mipSymmetryDetection);
      model->set(GRB_IntParam_RINS, mipParameters.mipRINS);
      model->set(GRB_IntParam_MIPFocus, mipParameters.mipMIPFocus);
      model->set(GRB_DoubleParam_Heuristics, mipParameters.mipHeuristics);
      model->set(GRB_DoubleParam_ImproveStartTime, mipParameters.mipImproveStartTime);
      model->set(GRB_DoubleParam_ImproveStartGap, mipParameters.mipImproveStartGap);
      if (mipParameters.mipDisablePresolve) model->set(GRB_IntParam_Presolve, 0);
      if (mipParameters.mipMINBPFORBID) {
        if (mipParameters.mipBranchingPriorities)
          model->set("GURO_PAR_MINBPFORBID", "5");  // TODO: add a parameter for that
        else {
          cout << "Branching priority was not enabled. Parameter MINBP_FORBID "
                  "has no effect."
               << endl;
        }
      }
      model->setCallback(&trajectory);
    } catch (GRBException e) {
      cout << "Error during model initialization." << endl;
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
      exit(0);
    } catch (...) {
      cout << "Error during model initialization." << endl;
      exit(0);
    }
    solved = false;
  }

  virtual ~BasicModel() {
    delete env;
    delete model;
    delete presolve;
    delete relaxed_model;
  }

  void add_variables() {
    add_variables_a();
    add_variables_y();
    add_variables_r();
    model->update();
  }

  void add_constraints() {
    add_constraints_C1();
    add_constraints_C2();
    add_constraints_C3();
    add_constraints_C4();
    add_constraints_C5();
    model->update();
  }

  void set_branching_priorities() {}

  void apply_preprocessing_strategies() {}

  void load_initial_solution(Solution &S) {
    // a
    for (VERTEXID v : I.get_Vf()) a[v].set(GRB_DoubleAttr_Start, S.fire_arrival_time[v]);

    // y
    for (VERTEXID v : I.get_Vf()) {
      if (ff(S.fire_arrival_time[v]) < ff(I.get_optimization_horizon()))
        y[v].set(GRB_DoubleAttr_Start, 1);
      else
        y[v].set(GRB_DoubleAttr_Start, 0);
    }
    // r
    for (RESOURCE i : I.get_R())
      for (VERTEXID u : I.get_Vf()) r[{i, u}].set(GRB_DoubleAttr_Start, 0);
    for (RESOURCE i : I.get_R())
      for (VERTEXID u : S.protected_vertices[i]) r[{i, u}].set(GRB_DoubleAttr_Start, 1);
    model->update();
  }

  void get_solution(Solution &B) {
    B = Solution(B.I, true, true);
    for (RESOURCE i : I.get_R()) {
      vector<VERTEXID> protected_vertices;
      for (VERTEXID v : I.get_Vf())
        if (r[{i, v}].get(GRB_DoubleAttr_X) > 0.5) {
          protected_vertices.push_back(v);
        }
      Solution::allocate_resources(B, i, protected_vertices);
    }
  }

  void add_objective_function() {
    GRBLinExpr obj = 0;
    for (VERTEXID v : I.get_Vf()) obj += y[v];
    model->setObjective(obj, GRB_MINIMIZE);
    model->update();
  }

  void build() {
    try {
      add_variables();
      add_constraints();
      add_objective_function();
    } catch (GRBException e) {
      cout << "Error during model building." << endl;
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
      exit(0);
    } catch (...) {
      cout << "Error during model building." << endl;
      exit(0);
    }

    if (mipParameters.mipPreprocessing) apply_preprocessing_strategies();
    if (mipParameters.mipBranchingPriorities) set_branching_priorities();

    if (mipParameters.mipSolveRelaxation) {
      relaxed_model = new GRBModel(model->relax());
      model_handle = relaxed_model;
    } else
      model_handle = model;

    if (mipParameters.mipComputePresolve) {
      try {
        presolve = new GRBModel(model_handle->presolve());
      } catch (GRBException e) {
        cout << "Error during presolve" << endl;
        cout << "Error code = " << e.getErrorCode() << endl;
        cout << e.getMessage() << endl;
        exit(0);
      } catch (...) {
        cout << "Error during presolve" << endl;
        exit(0);
      }
    }
  }

  void solve() {
    trajectory.init();
    try {
      model_handle->optimize();
    } catch (GRBException e) {
      cout << "Error during optimization" << endl;
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
      exit(0);
    } catch (...) {
      cout << "Error during optimization" << endl;
      exit(0);
    }
    solved = true;
  }

  void write() {
    try {
      model->write(I.get_instance_id() + ".lp");
      if (presolve != nullptr) presolve->write("presolved_" + I.get_instance_id() + ".lp");
    } catch (GRBException e) {
      cout << "Error during model writing." << endl;
      cout << "Error code = " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
      exit(0);
    } catch (...) {
      cout << "Error during model writing." << endl;
      exit(0);
    }
  }

  string get_objv() {
    if (model_handle->get(GRB_IntAttr_SolCount) > 0)
      return to_string(model_handle->get(GRB_DoubleAttr_ObjVal));
    else
      return "-";
  }

  string get_lower_bound() { return to_string(model_handle->get(GRB_DoubleAttr_ObjBound)); }

  string get_elapsed_time() { return to_string(model_handle->get(GRB_DoubleAttr_Runtime)); }

  string get_mip_gap() {
    if (!mipParameters.mipSolveRelaxation && model_handle->get(GRB_IntAttr_SolCount) > 0)
      return to_string(model_handle->get(GRB_DoubleAttr_MIPGap));
    else
      return "-";
  }

  string get_node_count() {
    if (mipParameters.mipSolveRelaxation)
      return "-";
    else
      return to_string(model_handle->get(GRB_DoubleAttr_NodeCount));
  }

  string get_solution_count() { return to_string(model_handle->get(GRB_IntAttr_SolCount)); }

  string get_open_node_count() {
    if (mipParameters.mipSolveRelaxation)
      return "-";
    else
      return to_string(model_handle->get(GRB_DoubleAttr_OpenNodeCount));
  }

  string get_num_constr() { return to_string(model_handle->get(GRB_IntAttr_NumConstrs)); }

  string get_num_vars() { return to_string(model_handle->get(GRB_IntAttr_NumVars)); }

  string get_num_bin_vars() { return to_string(model_handle->get(GRB_IntAttr_NumBinVars)); }

  string get_num_nzs() { return to_string(model_handle->get(GRB_IntAttr_NumNZs)); }

  string get_max_coeff() { return to_string(model_handle->get(GRB_DoubleAttr_MaxCoeff)); }

  string get_min_coeff() { return to_string(model_handle->get(GRB_DoubleAttr_MinCoeff)); }

  string get_max_bound() { return to_string(model_handle->get(GRB_DoubleAttr_MaxBound)); }

  string get_min_bound() { return to_string(model_handle->get(GRB_DoubleAttr_MinBound)); }

  string get_max_obj_coeff() { return to_string(model_handle->get(GRB_DoubleAttr_MaxObjCoeff)); }

  string get_min_obj_coeff() { return to_string(model_handle->get(GRB_DoubleAttr_MinObjCoeff)); }

  string get_max_rhs() { return to_string(model_handle->get(GRB_DoubleAttr_MaxRHS)); }

  string get_min_rhs() { return to_string(model_handle->get(GRB_DoubleAttr_MinRHS)); }

  string presolve_get_num_constr() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_IntAttr_NumConstrs));
  }

  string presolve_get_num_vars() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_IntAttr_NumVars));
  }

  string presolve_get_num_bin_vars() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_IntAttr_NumBinVars));
  }

  string presolve_get_num_nzs() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_IntAttr_NumNZs));
  }

  string presolve_get_max_coeff() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MaxCoeff));
  }

  string presolve_get_min_coeff() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MinCoeff));
  }

  string presolve_get_max_bound() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MaxBound));
  }

  string presolve_get_min_bound() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MinBound));
  }

  string presolve_get_max_obj_coeff() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MaxObjCoeff));
  }

  string presolve_get_min_obj_coeff() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MinObjCoeff));
  }

  string presolve_get_max_rhs() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MaxRHS));
  }

  string presolve_get_min_rhs() {
    if (presolve == nullptr)
      return "-";
    else
      return to_string(presolve->get(GRB_DoubleAttr_MinRHS));
  }

  bool is_relaxation() { return mipParameters.mipSolveRelaxation; }

  vector<tuple<unsigned, double, double>> get_trajectory() { return trajectory.trajectory; }
};
