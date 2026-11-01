#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <vector>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "beam_search.hpp"
#include "common.hpp"
#include "cut_heuristic.hpp"
#include "mip.hpp"
#include "parameters.hpp"
#include "random_search.hpp"

const char Instance::NULL_DELAY;

int main(int argc, char *argv[]) {
  // 1) Parse input
  struct Parameters {
    // General
    string instance, output, initsol, alg;
    bool verbose, save_solution, print_solution;
    unsigned timelimit, seed, max_iterations, target;
    double memlimit;

    // MIP
    string mipModel;
    bool mipSolveRelaxation, mipBranchingPriorities, mipDisablePresolve, mipMINBPFORBID,
        mipComputePresolve, mipTuning, mipPreprocessing, mipSaveModel;
    int mipSymmetryDetection, mipRINS, mipMIPFocus;
    double mipImproveStartTime, mipImproveStartGap, mipHeuristics;

    // Iterated Beam Search
    bool ibsFirePerimenterThreshold, ibsManyTrials;
    unsigned ibsBeta, ibsEta, ibsZmax, ibsC;
    double ibsP, ibsPhat;
    ull ibsMaxBudget;

    // Cut Heuristic
    double cuthEpsilon;
    unsigned cuthSolverTime;
  };
  Parameters opt;
  po::options_description general("General options");
  general.add_options()("instance", po::value<string>(&opt.instance),
                        "Path to instance specification file.")(
      "alg", po::value<string>(&opt.alg)->default_value("mip"), "Algorithm.")(
      "verbose", po::bool_switch(&opt.verbose)->default_value(false), "Verbosity.")(
      "output", po::value<string>(&opt.output), "Output file.")(
      "target", po::value<unsigned>(&opt.target)->default_value(0), "Target objective value.")(
      "maxiter",
      po::value<unsigned>(&opt.max_iterations)->default_value(numeric_limits<unsigned>::max()),
      "Maximum number of iterations.")("timelimit",
                                       po::value<unsigned>(&opt.timelimit)->default_value(7200),
                                       "Maximum running time (in seconds).")(
      "memlimit", po::value<double>(&opt.memlimit)->default_value(numeric_limits<double>::max()),
      "Available memory.")("print_solution",
                           po::bool_switch(&opt.print_solution)->default_value(false),
                           "Print best-found solution.")(
      "save_solution", po::bool_switch(&opt.save_solution)->default_value(false),
      "Save best-found solution.")("tuning", po::bool_switch(&opt.mipTuning)->default_value(false),
                                   "Print only obj. value for tuning.")(
      "initial_solution", po::value<string>(&opt.initsol)->default_value(""),
      "Path to initial solution (JSON).")(
      "seed", po::value<unsigned>(&opt.seed)->default_value(123), "Seed value.");
  po::options_description mip("MIP");
  mip.add_options()("mipModel", po::value<string>(&opt.mipModel)->default_value("basic"),
                    "MIP model.")("mipMINBPFORBID",
                                  po::bool_switch(&opt.mipMINBPFORBID)->default_value(false),
                                  "Forbid presolve to remove variables r_i_v and s_i_v.")(
      "mipDisablePresolve", po::bool_switch(&opt.mipDisablePresolve)->default_value(false),
      "Disable presolve.")("mipBranchingPriorities",
                           po::bool_switch(&opt.mipBranchingPriorities)->default_value(false),
                           "Assign branching priorities to variables.")(
      "mipSaveModel", po::bool_switch(&opt.mipSaveModel)->default_value(false),
      "Write model to an LP file.")(
      "mipPreprocessing", po::bool_switch(&opt.mipPreprocessing)->default_value(false),
      "Fix the value of some variables (when possible) during preprocessing.")(
      "mipSolveRelaxation", po::bool_switch(&opt.mipSolveRelaxation)->default_value(false),
      "Compute model relaxation.")("mipComputePresolve",
                                   po::bool_switch(&opt.mipComputePresolve)->default_value(false),
                                   "Compute presolved model.")(
      "mipSymmetryDetection", po::value<int>(&opt.mipSymmetryDetection)->default_value(-1),
      "Gurobi's symmetry detection parameter.")(
      "mipRINS", po::value<int>(&opt.mipRINS)->default_value(-1), "Gurobi'RINS parameter.")(
      "mipImproveStartTime",
      po::value<double>(&opt.mipImproveStartTime)->default_value(numeric_limits<double>::max()),
      "Time to start improving the solution (Gurobi).")(
      "mipImproveStartGap", po::value<double>(&opt.mipImproveStartGap)->default_value(0.0),
      "Gap to start improving the solution (Gurobi).")(
      "mipMIPFocus", po::value<int>(&opt.mipMIPFocus)->default_value(0),
      "Gurobi's MIP focus parameter.")("mipHeuristics",
                                       po::value<double>(&opt.mipHeuristics)->default_value(0.05),
                                       "Gurobi's Heuristics parameter.");
  po::options_description ibs("Iterated Beam Search");
  ibs.add_options()("ibsFirePerimeterThreshold",
                    po::value<bool>(&opt.ibsFirePerimenterThreshold)->default_value(true),
                    "Activate fire perimeter heuristic during allocation expansion.")(
      "ibsManyTrials", po::value<bool>(&opt.ibsManyTrials)->default_value(true),
      "Number of trials is proportional to fire perimeter size (standard "
      "algorithm).")("ibsP", po::value<double>(&opt.ibsP)->default_value(0.5),
                     "Probability of picking an element of N.")(
      "ibsBeta", po::value<unsigned>(&opt.ibsBeta)->default_value(50),
      "Number of starting nodes at each level.")(
      "ibsEta", po::value<unsigned>(&opt.ibsEta)->default_value(70), "Number of expansions.")(
      "ibsC", po::value<unsigned>(&opt.ibsC)->default_value(30),
      "Multiplier for the number of iterations performed by Step.")(
      "ibsPhat", po::value<double>(&opt.ibsPhat)->default_value(1),
      "Transition instant as a percentage of the free burning time.")(
      "ibsZmax", po::value<unsigned>(&opt.ibsZmax)->default_value(3),
      "Maximum value for z (only useful when ibsFirePerimenterThreshold is "
      "true).")("ibsMaxBudget",
                po::value<ull>(&opt.ibsMaxBudget)->default_value(numeric_limits<ull>::max()),
                "Maximum number of subtree updates.");
  po::options_description cuth("Cut Heuristic");
  cuth.add_options()("cuthEpsilon", po::value<double>(&opt.cuthEpsilon)->default_value(0.01),
                     "Maximum difference between the lower and upper bounds "
                     "during binary search.")(
      "cuthSolverTime", po::value<unsigned>(&opt.cuthSolverTime)->default_value(300),
      "Maximum time for the solver to run (in seconds).");
  general.add(ibs).add(mip).add(cuth);
  po::positional_options_description pod;
  pod.add("instance", 1);
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(general).positional(pod).run(), vm);
  po::notify(vm);
  if (vm.count("help")) {
    cout << general << endl;
    return 0;
  }
  if (!vm.count("instance")) {
    cerr << "Please provide an input instance." << endl;
    cout << general << endl;
    return 1;
  }

  // 2) Load instance
  Instance I(opt.instance);

  // 3) Create algorithm
  Algorithm *alg;
  GeneralParameters generalParameters;
  MipParameters mipParameters;
  BeamSearchParameters ibsParameters;
  CutHeuristicParameters cuthParameters;
  generalParameters.withSeed(opt.seed)
      .withTimelimit(opt.timelimit)
      .withMemLimit(opt.memlimit)
      .withMaxIterations(opt.max_iterations)
      .withTargetObjv(opt.target)
      .withVerbosity(opt.verbose)
      .withTuning(opt.mipTuning);
  mipParameters.withModel(opt.mipModel)
      .withPreprocessing(opt.mipPreprocessing)
      .withSolveRelaxation(opt.mipSolveRelaxation)
      .withSaveModel(opt.mipSaveModel)
      .withBranchingPriorities(opt.mipBranchingPriorities)
      .withDisablePresolve(opt.mipDisablePresolve)
      .withMINBPFORBID(opt.mipMINBPFORBID)
      .withComputePresolve(opt.mipComputePresolve)
      .withSymmetryDetection(opt.mipSymmetryDetection)
      .withRINS(opt.mipRINS)
      .withImproveStartTime(opt.mipImproveStartTime)
      .withImproveStartGap(opt.mipImproveStartGap)
      .withMIPFocus(opt.mipMIPFocus)
      .withHeuristics(opt.mipHeuristics);
  ibsParameters.withBeta(opt.ibsBeta)
      .withEta(opt.ibsEta)
      .withP(opt.ibsP)
      .withZmax(opt.ibsZmax)
      .withC(opt.ibsC)
      .withPhat(opt.ibsPhat)
      .withMaxBudget(opt.ibsMaxBudget)
      .withFirePerimenterThreshold(opt.ibsFirePerimenterThreshold)
      .withManyTrials(opt.ibsManyTrials);
  cuthParameters.withEpsilon(opt.cuthEpsilon).withSolverTime(opt.cuthSolverTime);
  if (opt.alg == "mip")
    alg = new MIP(I, generalParameters, mipParameters);
  else if (opt.alg == "ibs")
    alg = new BeamSearch(I, generalParameters, ibsParameters);
  else if (opt.alg == "cuth")
    alg = new CutHeuristic(I, generalParameters, cuthParameters);
  else if (opt.alg == "rs")
    alg = new RandomSearch(I, generalParameters);
  else {
    cout << "Unknown algorithm: " << opt.alg << endl;
    exit(1);
  }

  // 3.1) Load initial solution (if any)
  if (opt.initsol.size() > 0) alg->load_initial_solution(opt.initsol);

  // 4) Run algorithm
  alg->run();

  // 5) Save best solution
  if (opt.save_solution) {
    ofstream outfile("sol_" + opt.alg + "_" + I.get_instance_id() + "_" + to_string(opt.seed) +
                     ".json");
    alg->write_solution(outfile);
  }

  // 6) Print best solution
  if (opt.print_solution) {
    ofstream outfile("fig_" + opt.alg + "_" + I.get_instance_id() + "_" + to_string(opt.seed) +
                     ".txt");
    alg->print_solution(outfile);
  }

  // 7) Print statistics
  ofstream outfile(opt.output, ios_base::app);
  alg->write_statistics(outfile);
  if (opt.verbose) {
    alg->write_statistics(cout);
  }

  // 8) Clean up
  delete alg;
}
