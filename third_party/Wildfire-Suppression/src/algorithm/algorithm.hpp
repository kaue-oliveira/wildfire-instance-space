#pragma once

#include <ostream>
using namespace std;

#include "solution.hpp"

class Algorithm {
 public:
  virtual ~Algorithm() = default;
  virtual void run() = 0;                      // Run the algorithm
  virtual void get_solution(Solution &S) = 0;  // Get the solution found by the algorithm
  virtual void load_initial_solution(
      const string &file) = 0;                      // Load an initial solution from a JSON file
  virtual void write_statistics(ostream &out) = 0;  // Write statistics collected during the
                                                    // algorithm execution to an output stream
  virtual void write_solution(ostream &out) = 0;  // Write the best solution found by the algorithm
                                                  // to an output stream
  virtual void print_solution(ostream &out) = 0;  // Print the best solution found by the algorithm
                                                  // to an output stream
};