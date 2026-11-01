#pragma once

#include <vector>
using namespace std;

#include "common.hpp"
#include "gurobi_c++.h"
#include "instance.hpp"
#include "solution.hpp"

class CutModel {
 private:
  Instance& I;
  Solution& S;
  GRBEnv env;
  GRBModel model;
  map<VERTEXID, GRBVar> x;  // Cut set variables
  map<VERTEXID, GRBVar> y;  // Reachability variables

 public:
  CutModel(Solution& _S, TIME threshold, CAPACITY num_resources, unsigned timelimit)
      : I(_S.I), S(_S), env(), model(env) {
    // Initialize model
    model.set(GRB_IntParam_OutputFlag, false);
    model.set(GRB_DoubleParam_TimeLimit, timelimit);
    model.set(GRB_IntParam_Threads, 1);
    // Find source vertices
    set<VERTEXID> sources;
    for (VERTEXID v : I.get_V())
      if (S.fire_arrival_time[v] < threshold) sources.insert(v);
    // Create variables
    for (VERTEXID v : I.get_V()) {
      x[v] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + to_string(v)); // cut 
      y[v] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y_" + to_string(v)); // reachability
    }
    // Limit the cut size to num_resources
    GRBLinExpr cut_size_constraint = 0;
    for (VERTEXID v : I.get_V()) {
      cut_size_constraint += x[v];
    }
    model.addConstr(cut_size_constraint <= num_resources, "CutSizeConstraint");
    // Source vertices in S must be reachable
    for (VERTEXID s : sources) {
      model.addConstr(y[s] == 1, "SourceReachability_" + to_string(s));
    }
    // Reachability through arcs
    for (VERTEXID u : I.get_V()) {
      for (auto& [v, t_uv] : I.get_outgoing_arcs(u)) {
        if (y.find(v) != y.end())
          model.addConstr(y[u] <= y[v] + x[u],
                          "ReachabilityPropagation_" + to_string(u) + "_" + to_string(v));
      }
    }
    // Ensure source vertices cannot be in the cut set
    for (VERTEXID s : sources) {
      model.addConstr(x[s] == 0, "SourceNotInCut_" + to_string(s));
    }
    // Objective: Minimize sum of y_i (reachable vertices)
    GRBLinExpr objective = 0;
    for (VERTEXID v : I.get_V()) {
      objective += y[v];
    }
    model.setObjective(objective, GRB_MINIMIZE);
    model.update();
  }

  bool optimize(vector<VERTEXID>& P) {
    model.optimize();
    P.clear();
    if (model.get(GRB_IntAttr_SolCount) > 0) {
      for (auto& [v, var] : x)
        if (x[v].get(GRB_DoubleAttr_X) > 0.5) P.push_back(v);
      return true;
    } else {
      return false;
    }
  }
};
