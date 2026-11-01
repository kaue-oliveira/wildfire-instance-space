#pragma once

#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <ranges>
#include <regex>
#include <vector>
using namespace std;
namespace views = std::views;

#include "common.hpp"

struct Vertex {
  vector<pair<VERTEXID, TIME>> incoming;
  vector<pair<VERTEXID, TIME>> outgoing;
  Vertex() = default;
  void add_incoming_arc(VERTEXID v, TIME t) { incoming.push_back({v, t}); }
  void add_outgoing_arc(VERTEXID v, TIME t) { outgoing.push_back({v, t}); }
};

class Digraph {
 private:
  VERTEXID vertex_id_counter = 0;
  vector<Vertex> vertices;
  map<pair<VERTEXID, VERTEXID>, TIME> arc_costs;

  set<VERTEXID> blocked_vertex;
  set<pair<VERTEXID, VERTEXID>> blocked_arc;

 public:
  Digraph() = default;

  Digraph(VERTEXID num_nodes) {
    for (VERTEXID i = 0; i < num_nodes; i++) {
      vertices.push_back(Vertex());
      vertex_id_counter++;
    }
  }

  void add_arc(VERTEXID source, VERTEXID destination, TIME time) {
    vertices[source].outgoing.push_back({destination, time});
    vertices[destination].incoming.push_back({source, time});
    arc_costs[{source, destination}] = time;
  }

  TIME get_arc_cost(VERTEXID u, VERTEXID v) { return arc_costs[{u, v}]; }

  auto get_outgoing_arcs(VERTEXID u) {
    bool is_blocked_u = blocked_vertex.find(u) != blocked_vertex.end();
    return vertices[u].outgoing |
           std::views::filter([this, u, is_blocked_u](pair<VERTEXID, TIME> arc) {
             bool is_blocked_v = blocked_vertex.find(arc.first) != blocked_vertex.end();
             bool is_blocked_arc = blocked_arc.find({u, arc.first}) != blocked_arc.end();
             return !is_blocked_u && !is_blocked_v && !is_blocked_arc;
           });
  }

  auto get_incoming_arcs(VERTEXID v) {
    bool is_blocked_v = blocked_vertex.find(v) != blocked_vertex.end();
    return vertices[v].incoming |
           std::views::filter([this, v, is_blocked_v](pair<VERTEXID, TIME> arc) {
             bool is_blocked_u = blocked_vertex.find(arc.first) != blocked_vertex.end();
             bool is_blocked_arc = blocked_arc.find({arc.first, v}) != blocked_arc.end();
             return !is_blocked_v && !is_blocked_u && !is_blocked_arc;
           });
  }

  // The methods below are only used by Yen's algorithm to compute the kth
  // shortest path.
  void restore_vertex(VERTEXID u) { blocked_vertex.erase(u); }

  void restore_arc(VERTEXID u, VERTEXID v) { blocked_arc.erase({u, v}); }

  void block_vertex(VERTEXID u) { blocked_vertex.insert(u); }

  void block_arc(VERTEXID u, VERTEXID v) { blocked_arc.insert({u, v}); }

  void restore_vertices() { blocked_vertex.clear(); }

  void restore_arcs() { blocked_arc.clear(); }
};
