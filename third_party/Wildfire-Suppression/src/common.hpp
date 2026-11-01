#pragma once

using ul = unsigned long;
using ull = unsigned long long;
using VALUE = double;
using CAPACITY = unsigned;
using DISTANCE = double;
using RESOURCE = unsigned;
using VERTEXID = unsigned;
using TIME = double;
using DELAYID = char;

constexpr TIME INF_TIME = numeric_limits<TIME>::max();
constexpr VALUE INF_VALUE = numeric_limits<VALUE>::max();
constexpr VERTEXID INF_VERTEXID = numeric_limits<VERTEXID>::max();
constexpr DISTANCE INF_DISTANCE = numeric_limits<DISTANCE>::max();
constexpr CAPACITY INF_CAPACITY = numeric_limits<CAPACITY>::max();
constexpr RESOURCE INF_RESOURCE = numeric_limits<RESOURCE>::max();
constexpr double INF_double = numeric_limits<double>::max();

// https://github.com/AlexGliesch/critical-coloring/blob/master/src/util.h
struct ff {
  static constexpr double EPS = 1e-4;
  static bool fEq(double a, double b) { return abs(a - b) < EPS; }
  ff() = default;
  ff(double v) : v(v) {}
  bool operator==(const ff& o) const { return fEq(v, o.v); }
  bool operator!=(const ff& o) const { return !fEq(v, o.v); }
  bool operator<=(const ff& o) const { return v < o.v || fEq(v, o.v); }
  bool operator>=(const ff& o) const { return v > o.v || fEq(v, o.v); }
  bool operator<(const ff& o) const { return v + EPS < o.v; }
  bool operator>(const ff& o) const { return o < *this; }
  double v;
};
