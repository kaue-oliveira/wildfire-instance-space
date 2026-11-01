#pragma once

#include <fstream>
#include <regex>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>
using namespace std;

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <boost/multi_array.hpp>
using namespace boost;

#include "digraph.hpp"

struct Instance {
  // All time-like quantities share this exact fixed-point scale. Historical
  // integral inputs use 1; inputs with at most two decimal places use 100.
  // The search still uses integer arithmetic, with no truncation of decimals.
  unsigned time_scale = 1;
  unsigned H;                  // Optimization horizon
  unsigned n;                  // Number of nodes
  unsigned ign;                // Ignition node
  unsigned Delta;              // Delay

  vector<pair<unsigned, unsigned>> R; // Number of resources that become available at instant k
  vector<unsigned> T; // Time instants at which resources are available
  Digraph G; // Graph

  vector<set<NODE_ID>> N; // Immediate neighborhood
  vector<set<NODE_ID>> Nstar; // Extended neighborhood

  static long double numeric_value(const json& value) {
    if (!value.is_number())
      throw invalid_argument("A WSP time or edge weight must be a JSON number");
    const long double number = value.get<long double>();
    if (!isfinite(number) || number < 0)
      throw invalid_argument("A WSP time or edge weight must be finite and nonnegative");
    return number;
  }

  static long double resource_time(const string& value) {
    size_t consumed = 0;
    const long double number = stold(value, &consumed);
    if (consumed != value.size() || !isfinite(number) || number < 0)
      throw invalid_argument("Invalid resource time; JSON decimal separator must be '.'");
    return number;
  }

  static bool has_fraction(long double number) {
    return number != roundl(number);
  }

  unsigned scaled_time(long double number) const {
    if (!isfinite(number) || number < 0)
      throw invalid_argument("A WSP time or edge weight must be finite and nonnegative");
    const long double scaled = number * time_scale;
    const long double nearest = roundl(scaled);
    const long double parse_tolerance =
        4 * numeric_limits<double>::epsilon() * max(1.0L, fabsl(scaled));
    if (fabsl(scaled - nearest) > parse_tolerance)
      throw invalid_argument("WSP temporal input is not representable in exact hundredths");
    if (nearest >= static_cast<long double>(numeric_limits<int>::max()))
      throw overflow_error("Scaled WSP temporal input exceeds the search's signed INF sentinel");
    return static_cast<unsigned>(nearest);
  }

  unsigned scaled_time(const json& value) const {
    return scaled_time(numeric_value(value));
  }

  string format_time(unsigned value) const {
    if (time_scale == 1)
      return to_string(value);
    ostringstream out;
    out << value / time_scale << '.' << setw(2) << setfill('0') << value % time_scale;
    return out.str();
  }

  json json_time(unsigned value) const {
    if (time_scale == 1 || value == static_cast<unsigned>(numeric_limits<int>::max()))
      return value;
    return static_cast<double>(value) / time_scale;
  }

  void build_neighborhoods() {
      N = vector<set<NODE_ID>>(G.get_number_of_nodes());
      Nstar = vector<set<NODE_ID>>(G.get_number_of_nodes());
      for(const COORD& n_c : G.get_list_of_coords()) {
        unsigned nid = G.get_node_id(n_c);
        unsigned x = n_c.first, y = n_c.second;
        COORD n1 = {x == 0 ? 0 : x - 1, y};
        COORD n2 = {x+1, y};
        COORD n3 = {x, y == 0 ? 0 : y - 1};
        COORD n4 = {x, y + 1};
        for(COORD ni : {n1, n2, n3, n4})
          if(ni != n_c && G.is_valid(ni)) {
            N[nid].insert(G.get_node_id(ni));
            Nstar[nid].insert(G.get_node_id(ni));
          }
        COORD n5 = {x == 0 ? 0 : x - 1, y == 0 ? 0 : y - 1};
        COORD n6 = {x == 0 ? 0 : x - 1, y + 1};
        COORD n7 = {x + 1, y == 0 ? 0 : y - 1};
        COORD n8 = {x + 1, y + 1};
        for(COORD ni : {n5, n6, n7, n8})
          if(ni != n_c && G.is_valid(ni)) 
            Nstar[nid].insert(G.get_node_id(ni));   
    } 
  }

  void read_instance(const string &instance_file) {
    ifstream f(instance_file);
    if (!f.good()) {
      cerr << "Could not open instance specification file: " << instance_file << endl;
      exit(1);
    }
    json instance = json::parse(f);
    f.close();

    // Detect the required scale before reading any temporal value. A fixed
    // factor of 100 exactly represents every decimal in Benchmark74's 19
    // supplementary Harris files; unsupported finer precision is rejected.
    time_scale = 1;
    if (has_fraction(numeric_value(instance.at("ArrivalTimeTarget"))) ||
        has_fraction(numeric_value(instance.at("Delay"))))
      time_scale = 100;
    for (const auto& res : instance.at("ResAtTime").items())
      if (has_fraction(resource_time(res.key())))
        time_scale = 100;
    for (const auto& edge : instance.at("Arcs").items())
      if (has_fraction(numeric_value(edge.value())))
        time_scale = 100;

    // Optimization Horizon
    H = scaled_time(instance.at("ArrivalTimeTarget"));
    R = vector<pair<unsigned, unsigned>>();
    T.clear();

    // Resources
    for (const auto &res : instance["ResAtTime"].items()) {
      unsigned instant = scaled_time(resource_time(res.key()));
      unsigned quantity = res.value();
      R.emplace_back(instant, quantity);
    }
    sort(R.begin(), R.end(), [](pair<unsigned, unsigned> a, pair<unsigned, unsigned> b){return a.first < b.first;});
    for (size_t index = 1; index < R.size(); index++)
      if (R[index - 1].first == R[index].first)
        throw invalid_argument("Two resource times coincide at the selected exact scale");
    if (R.empty())
      throw invalid_argument("The IBS requires at least one resource-release instant");
    for(const auto& entry : R)
      T.push_back(entry.first);

    // Delta
    Delta = scaled_time(instance.at("Delay"));

    // Nodes
    G = Digraph();
    for (const auto &n : instance["Nodes"]) {
      unsigned i = n.at(0).get<unsigned>();
      unsigned j = n.at(1).get<unsigned>();
      G.add_node(MAKE_COORD(i, j));
    }
    n = G.get_number_of_nodes();

    // Ignition node
    unsigned ign_x = instance["Ignitions"].at(0).at(0).get<unsigned>();
    unsigned ign_y = instance["Ignitions"].at(0).at(1).get<unsigned>();
    ign = G.get_node_id(MAKE_COORD(ign_x, ign_y));

    // Edges
    regex rgx("\\((-?\\d+), (-?\\d+)\\)");
    for (const auto &edge : instance["Arcs"].items()) {
      const string coords = edge.key();
      unsigned weight = scaled_time(edge.value());

      vector<int> values;
      sregex_iterator iterator(coords.begin(), coords.end(), rgx);
      sregex_iterator endIterator;
      while (iterator != endIterator) {
        smatch match = *iterator;
        int firstValue = stoi(match[1]);
        int secondValue = stoi(match[2]);
        values.push_back(firstValue);
        values.push_back(secondValue);
        ++iterator;
      }
      COORD c1 = MAKE_COORD(values[0], values[1]);
      COORD c2 = MAKE_COORD(values[2], values[3]);
      G.add_edge(c1, c2, weight);
    }

    // The original search uses signed INT_MAX as its infinity sentinel and
    // unsigned 32-bit sums for survival time. Reject cases that could wrap.
    unsigned max_weight = 0;
    for (const auto& edge : instance.at("Arcs").items())
      max_weight = max(max_weight, scaled_time(edge.value()));
    if (static_cast<unsigned long long>(n) * (max_weight + static_cast<unsigned long long>(Delta)) >=
            static_cast<unsigned long long>(numeric_limits<int>::max()) ||
        static_cast<unsigned long long>(n) * H >= numeric_limits<unsigned>::max())
      throw overflow_error("Scaled WSP instance exceeds the IBS integer bounds");

    build_neighborhoods();
  }

  const vector<Edge*> &get_outgoing_edges(NODE_ID node_id) {
    return G.get_outgoing_edges(node_id);
  }

  const vector<Edge*> &get_incoming_edges(NODE_ID node_id) {
    return G.get_incoming_edges(node_id);
  }

  unsigned get_number_of_edges() { return G.get_number_of_edges(); }
  
  unsigned get_edge_cost(NODE_ID pred, NODE_ID succ){
    return G.get_edge_cost(pred, succ);
  }

  unsigned alpha(unsigned t) {
    if(t >= T.back())
      return H;
    else
      return *upper_bound(T.begin(), T.end(), t);
  }

  const set<NODE_ID>& get_extended_neighborhood(NODE_ID nid){
    return Nstar[nid];
  }

  const set<NODE_ID>& get_neighborhood(NODE_ID nid){
    return N[nid];
  }

  string get_node_signature(NODE_ID id) {
      return G.get_node_signature(id);
  }
};
