#pragma once

#include <chrono>
using namespace std::chrono;

class Clock {
 private:
  steady_clock::time_point start;
  bool is_running;

 public:
  Clock(bool _start) {
    if (_start) start_clock();
  };

  void start_clock() {
    is_running = true;
    start = steady_clock::now();
  }

  unsigned now() { return duration_cast<seconds>(steady_clock::now() - start).count(); }
};