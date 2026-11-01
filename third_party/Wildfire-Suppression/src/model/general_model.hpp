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

class GeneralModel : public Model {
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
  map<VERTEXID, GRBVar> d;
  map<VERTEXID, GRBVar> Delta;
  map<VERTEXID, GRBVar> y;
  map<pair<VERTEXID, VERTEXID>, GRBVar> p;
  map<pair<RESOURCE, VERTEXID>, GRBVar> s;
  map<pair<RESOURCE, VERTEXID>, GRBVar> r;
  vector<VERTEXID> M;
  // Stats
  bool solved;

  void add_variables_a() {
    for (VERTEXID v : I.get_Vf())
      a[v] = model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "a_" + to_string(v));
  }

  void add_variables_Delta() {
    for (VERTEXID v : I.get_Vf())
      Delta[v] = model->addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, "Delta_" + to_string(v));
  }

  void add_variables_y() {
    for (VERTEXID v : I.get_Vf())
      y[v] = model->addVar(0.0, 1.0, 1.0, GRB_BINARY, "y_" + to_string(v));
  }

  void add_variables_d() {
    for (RESOURCE i : I.get_R())
      d[i] = model->addVar(I.get_release_time(i), I.get_optimization_horizon(), 0.0, GRB_CONTINUOUS,
                           "d_" + to_string(i));
  }

  void add_variables_p() {
    for (VERTEXID v : I.get_Vf()) {
      if (!I.inI(v))
        for (auto &[u, t_uv] : I.get_incoming_arcs(v))
          if (I.inVf(u))
            p[{u, v}] =
                model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "p_" + to_string(u) + "_" + to_string(v));
    }
  }

  void add_variables_s() {
    for (RESOURCE i : I.get_R())
      if (I.get_range(i) < INF_DISTANCE)
        for (VERTEXID v : I.get_Vb(i))
          s[{i, v}] =
              model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "s_" + to_string(i) + "_" + to_string(v));
  }

  void add_variables_r() {
    for (RESOURCE i : I.get_R())
      for (VERTEXID v : I.get_Vp(i))
        r[{i, v}] =
            model->addVar(0.0, 1.0, 0.0, GRB_BINARY, "r_" + to_string(i) + "_" + to_string(v));
  }

  void add_constraints_fire_propagation_and_delay() {
    // Upper bound on fire arrival time
    for (VERTEXID v : I.get_Vf()) {
      if (I.inI(v))
        model->addConstr(a[v] == 0, "C1.a." + to_string(v));
      else
        for (auto &[u, t_uv] : I.get_incoming_arcs(v))
          if (I.inVf(u))
            model->addConstr(a[v] <= a[u] + t_uv + Delta[u],
                             "C1.b" + to_string(v) + "_" + to_string(u));
    }
    // Lower bound on fire arrival time
    for (VERTEXID v : I.get_Vf()) {
      if (I.inI(v)) continue;
      for (auto &[u, t_uv] : I.get_incoming_arcs(v))
        if (I.inVf(u))
          model->addConstr(a[v] >= a[u] + t_uv + Delta[u] - M[u] * (1 - p[{u, v}]),
                           "C13." + to_string(v) + "_" + to_string(u));
    }
    // Predecessor constraints
    for (VERTEXID v : I.get_Vf()) {
      if (I.inI(v)) continue;
      GRBLinExpr sum_p_u_v = 0;
      for (auto &[u, t_uv] : I.get_incoming_arcs(v))
        if (I.inVf(u)) sum_p_u_v += p[{u, v}];
      model->addConstr(sum_p_u_v == 1, "C14." + to_string(v));
    }
    // Delay added to vertex v
    for (VERTEXID v : I.get_Vf()) {
      GRBLinExpr expr = 0;
      for (RESOURCE i : I.get_R()) {
        TIME delay = I.get_resource_delay(i);
        if (I.inVp(v, i)) expr += delay * r[{i, v}];
      }
      model->addConstr(Delta[v] == expr, "C4." + to_string(v));
    }
  }

  void add_constraints_base_location_and_range() {
    // Each resource has at most one base location
    for (RESOURCE i : I.get_R()) {
      DISTANCE range = I.get_range(i);
      if (range == INF_DISTANCE) continue;
      GRBLinExpr sum_s_i_v = 0;
      for (VERTEXID v : I.get_Vb(i)) sum_s_i_v += s[{i, v}];
      model->addConstr(sum_s_i_v <= 1, "C9." + to_string(i));
    }
    // If vertex v received resource i, then the base location of i must be
    // nearby v
    for (RESOURCE i : I.get_R()) {
      DISTANCE range = I.get_range(i);
      if (range == INF_DISTANCE) continue;
      for (VERTEXID v : I.get_Vp(i)) {
        GRBLinExpr sum_s_i_u = 0;
        for (VERTEXID u : I.get_Vb(i))
          if (ff(I.distance(u, v)) <= ff(range)) sum_s_i_u += s[{i, u}];
        model->addConstr(r[{i, v}] <= sum_s_i_u, "C8." + to_string(i) + "_" + to_string(v));
      }
    }
  }

  void add_constraints_resource_capacity() {
    // Each resource cannot protect more vertices than its capacity
    for (RESOURCE i : I.get_R()) {
      CAPACITY capacity = I.get_capacity(i);
      GRBLinExpr sum_r_i_v = 0;
      for (VERTEXID v : I.get_Vp(i)) sum_r_i_v += r[{i, v}];
      model->addConstr(sum_r_i_v <= capacity, "C2." + to_string(i));
    }
    // Each flammable vertex can receive at most one resource
    for (VERTEXID v : I.get_Vf()) {
      GRBLinExpr sum_r_i_v = 0;
      for (RESOURCE i : I.get_R())
        if (I.inVp(v, i)) sum_r_i_v += r[{i, v}];
      model->addConstr(sum_r_i_v <= 1, "C3." + to_string(v));
    }
  }

  void add_constraints_deployment_time() {
    // No constraints needed, deployment time is already set in
    // add_variables_d()
  }

  void add_constraints_safety_interval() {
    // Safety constraints regarding the base location of resource i
    TIME H = I.get_optimization_horizon();
    for (RESOURCE i : I.get_R()) {
      if (I.get_range(i) == INF_DISTANCE) continue;
      TIME zb = I.get_base_safety_time(i);
      DISTANCE db = I.get_base_safety_distance(i);
      TIME M = H + zb;
      for (VERTEXID v : I.get_Vb(i))
        for (VERTEXID u : I.get_Vf())
          if (I.distance(v, u) <= db)
            model->addConstr(-M * (1 - s[{i, v}]) + d[i] + zb <= a[u],
                             "C15." + to_string(i) + "_" + to_string(v) + "_" + to_string(u));
    }
    // Safety constraints regarding the vertices protected by resource i
    for (RESOURCE i : I.get_R()) {
      TIME zp = I.get_protection_safety_time(i);
      DISTANCE dp = I.get_protection_safety_distance(i);
      TIME M = H + zp;
      for (VERTEXID v : I.get_Vp(i))
        for (VERTEXID u : I.get_Vf())
          if (I.distance(v, u) <= dp)
            model->addConstr(-M * (1 - r[{i, v}]) + d[i] + zp <= a[u],
                             "C15." + to_string(i) + "_" + to_string(v) + "_" + to_string(u));
    }
  }

  void add_constraints_expiration_interval() {
    // Expiration times
    for (RESOURCE i : I.get_R()) {
      TIME release_time = I.get_release_time(i);
      TIME e = I.get_expiration_time(i);
      if (e < INF_TIME)
        for (VERTEXID v : I.get_Vp(i))
          model->addConstr(M[v] * (1 - r[{i, v}]) + d[i] + e >= a[v],
                           "C12." + to_string(i) + "_" + to_string(v));
      else
        model->addConstr(d[i] == release_time, "C12." + to_string(i));
    }
  }

  void add_constraints_burned_vertices() {
    TIME H = I.get_optimization_horizon();
    for (VERTEXID v : I.get_Vf()) {
      model->addConstr(y[v] >= 1 - a[v] / H, "C17." + to_string(v));
    }
  }

 public:
  GeneralModel(Instance &_I, GeneralParameters gp, MipParameters mp)
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
      model->set(GRB_DoubleParam_ImproveStartTime, mipParameters.mipImproveStartTime);
      model->set(GRB_DoubleParam_ImproveStartGap, mipParameters.mipImproveStartGap);
      if (mipParameters.mipDisablePresolve) model->set(GRB_IntParam_Presolve, 0);
      if (mipParameters.mipMINBPFORBID) {
        if (mipParameters.mipBranchingPriorities)
          model->set("GURO_PAR_MINBPFORBID", "5");
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

  virtual ~GeneralModel() {
    delete env;
    delete model;
    delete presolve;
    delete relaxed_model;
  }

  void compute_Big_M() {
    // Find max delta_i
    TIME max_delta = 0;
    for (RESOURCE i : I.get_R()) max_delta = max(max_delta, I.get_resource_delay(i));

    // Find maximum possible arrival time to each vertex
    M = vector<VERTEXID>(I.get_Vf().size());
    Solution S(I, true, true);
    for (VERTEXID v : I.get_Vf()) {
      if (I.inI(v))
        M[v] = 0;
      else {
        // Count number of vertices in the shortest path to v
        VERTEXID u = v;
        unsigned count = 0;
        while (u != S.predecessor[u]) {
          count++;
          u = S.predecessor[u];
        }
        M[v] = S.fire_arrival_time[v] + max_delta * count;
      }
    }
  }

  void add_variables() {
    add_variables_a();
    add_variables_Delta();
    add_variables_y();
    add_variables_d();
    add_variables_r();
    add_variables_s();
    add_variables_p();
    model->update();
  }

  void add_constraints() {
    add_constraints_fire_propagation_and_delay();
    add_constraints_base_location_and_range();
    add_constraints_resource_capacity();
    add_constraints_deployment_time();
    add_constraints_safety_interval();
    add_constraints_expiration_interval();
    add_constraints_burned_vertices();
    model->update();
  }

  void set_branching_priorities() {}

  void apply_preprocessing_strategies() {}

  void load_initial_solution(Solution &S) {
    // a, Delta
    for (VERTEXID v : I.get_Vf()) {
      a[v].set(GRB_DoubleAttr_Start, S.fire_arrival_time[v]);
      Delta[v].set(GRB_DoubleAttr_Start, S.get_vertex_delay(v));
      if (!I.inI(v)) {
        for (auto &[u, t_uv] : I.get_incoming_arcs(v)) p[{u, v}].set(GRB_DoubleAttr_Start, 0);
        p[{S.predecessor[v], v}].set(GRB_DoubleAttr_Start, 1);
      }
    }
    // y
    for (VERTEXID v : I.get_Vf()) {
      if (S.fire_arrival_time[v] < I.get_optimization_horizon())
        y[v].set(GRB_DoubleAttr_Start, 1);
      else
        y[v].set(GRB_DoubleAttr_Start, 0);
    }
    // r
    for (RESOURCE i : I.get_R()) {
      for (VERTEXID u : I.get_Vp(i)) r[{i, u}].set(GRB_DoubleAttr_Start, 0);
      if (I.get_range(i) < INF_DISTANCE)
        for (VERTEXID u : I.get_Vb(i)) s[{i, u}].set(GRB_DoubleAttr_Start, 0);
      d[i].set(GRB_DoubleAttr_Start, I.get_optimization_horizon());
    }
    for (RESOURCE i : I.get_R()) {
      for (VERTEXID u : S.protected_vertices[i]) r[{i, u}].set(GRB_DoubleAttr_Start, 1);
      if (I.get_range(i) < INF_DISTANCE) s[{i, S.base_location[i]}].set(GRB_DoubleAttr_Start, 1);
      d[i].set(GRB_DoubleAttr_Start, S.deployment_time[i]);
    }
    model->update();
  }

  void add_objective_function() {
    GRBLinExpr obj = 0;
    for (VERTEXID v : I.get_Vf()) obj += I.get_value(v) * y[v];
    model->setObjective(obj, GRB_MINIMIZE);
    model->update();
  }

  void get_solution(Solution &B) {
    B = Solution(B.I, true, true);
    for (RESOURCE i : I.get_R()) {
      for (VERTEXID v : I.get_Vp(i))
        if (r[{i, v}].get(GRB_DoubleAttr_X) > 0.5) B.allocate_resource(i, v);

      if (I.get_range(i) < INF_DISTANCE) {
        for (VERTEXID v : I.get_Vb(i))
          if (s[{i, v}].get(GRB_DoubleAttr_X) > 0.5) B.set_base_location(i, v);
      } else
        B.set_base_location(i, INF_VERTEXID);
      B.set_deployment_time(i, d[i].get(GRB_DoubleAttr_X));
    }
    B.update();
    Solution::check_feasibility(B);
  }

  void build() {
    try {
      compute_Big_M();
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
