#pragma once

#include "common.hpp"

struct BeamSearchParameters {
  bool ibsFirePerimenterThreshold, ibsManyTrials;
  unsigned ibsZmax, ibsC, ibsBeta, ibsEta;
  double ibsPhat, ibsP;
  ull ibsMaxBudget;

  BeamSearchParameters() {
    ibsFirePerimenterThreshold = true;
    ibsZmax = 0;
    ibsC = 0;
    ibsBeta = 0;
    ibsEta = 0;
    ibsPhat = 0;
    ibsP = 0;
    ibsMaxBudget = 0;
  }

  BeamSearchParameters &withManyTrials(bool v) {
    ibsManyTrials = v;
    return *this;
  }

  BeamSearchParameters &withFirePerimenterThreshold(bool v) {
    ibsFirePerimenterThreshold = v;
    return *this;
  }

  BeamSearchParameters &withZmax(unsigned v) {
    ibsZmax = v;
    return *this;
  }

  BeamSearchParameters &withC(unsigned v) {
    ibsC = v;
    return *this;
  }

  BeamSearchParameters &withBeta(unsigned v) {
    ibsBeta = v;
    return *this;
  }

  BeamSearchParameters &withEta(unsigned v) {
    ibsEta = v;
    return *this;
  }

  BeamSearchParameters &withPhat(double v) {
    ibsPhat = v;
    return *this;
  }

  BeamSearchParameters &withP(double v) {
    ibsP = v;
    return *this;
  }

  BeamSearchParameters &withMaxBudget(ull v) {
    ibsMaxBudget = v;
    return *this;
  }
};

struct CutHeuristicParameters {
  double cuthEpsilon;
  unsigned cuthSolverTime;

  CutHeuristicParameters() {
    cuthEpsilon = 0;
    cuthSolverTime = 0;
  }

  CutHeuristicParameters &withEpsilon(double v) {
    cuthEpsilon = v;
    return *this;
  }

  CutHeuristicParameters &withSolverTime(unsigned v) {
    cuthSolverTime = v;
    return *this;
  }
};

struct MipParameters {
  string mipModel;
  bool mipPreprocessing, mipSolveRelaxation, mipSaveModel, mipBranchingPriorities,
      mipDisablePresolve, mipComputePresolve, mipMINBPFORBID;
  int mipSymmetryDetection, mipRINS, mipMIPFocus;
  double mipImproveStartTime, mipImproveStartGap, mipHeuristics;

  MipParameters() {
    mipModel = "";
    mipRINS = -1;
    mipSymmetryDetection = -1;
    mipMIPFocus = 0;
    mipImproveStartTime = numeric_limits<double>::max();
    mipImproveStartGap = 0.0;
    mipHeuristics = 0.05;
    mipPreprocessing = false;
    mipSolveRelaxation = false;
    mipSaveModel = false;
    mipBranchingPriorities = false;
    mipDisablePresolve = false;
    mipMINBPFORBID = false;
    mipComputePresolve = false;
  }

  MipParameters &withPreprocessing(bool v) {
    mipPreprocessing = v;
    return *this;
  }

  MipParameters &withSolveRelaxation(bool v) {
    mipSolveRelaxation = v;
    return *this;
  }

  MipParameters &withSaveModel(bool v) {
    mipSaveModel = v;
    return *this;
  }

  MipParameters &withBranchingPriorities(bool v) {
    mipBranchingPriorities = v;
    return *this;
  }

  MipParameters &withDisablePresolve(bool v) {
    mipDisablePresolve = v;
    return *this;
  }

  MipParameters &withMINBPFORBID(bool v) {
    mipMINBPFORBID = v;
    return *this;
  }

  MipParameters &withComputePresolve(bool v) {
    mipComputePresolve = v;
    return *this;
  }

  MipParameters &withSymmetryDetection(int v) {
    mipSymmetryDetection = v;
    return *this;
  }

  MipParameters &withRINS(int v) {
    mipRINS = v;
    return *this;
  }

  MipParameters &withImproveStartTime(double v) {
    mipImproveStartTime = v;
    return *this;
  }

  MipParameters &withImproveStartGap(double v) {
    mipImproveStartGap = v;
    return *this;
  }

  MipParameters &withModel(string v) {
    mipModel = v;
    return *this;
  }

  MipParameters &withMIPFocus(int v) {
    mipMIPFocus = v;
    return *this;
  }

  MipParameters &withHeuristics(double v) {
    mipHeuristics = v;
    return *this;
  }
};

struct GeneralParameters {
  bool verbose, tuning;
  unsigned seed, timelimit, max_iterations;
  double memlimit;
  VALUE target_objv;

  GeneralParameters() {
    seed = 0;
    timelimit = 0;
    max_iterations = 0;
    target_objv = 0;
    memlimit = 0;
    verbose = false;
    tuning = false;
  }

  GeneralParameters &withSeed(unsigned v) {
    seed = v;
    return *this;
  }

  GeneralParameters &withTimelimit(unsigned v) {
    timelimit = v;
    return *this;
  }

  GeneralParameters &withMaxIterations(unsigned v) {
    max_iterations = v;
    return *this;
  }

  GeneralParameters &withTargetObjv(unsigned v) {
    target_objv = v;
    return *this;
  }

  GeneralParameters &withVerbosity(bool v) {
    verbose = v;
    return *this;
  }

  GeneralParameters &withTuning(bool v) {
    tuning = v;
    return *this;
  }

  GeneralParameters &withMemLimit(double v) {
    memlimit = v;
    return *this;
  }
};