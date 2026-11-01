#pragma once

#include <iostream>
#include <vector>
using namespace std;

#include <boost/heap/binomial_heap.hpp>
using namespace boost::heap;

#include "common.hpp"

class MyHeap {
 private:
  typedef struct HeapNode {
    VERTEXID vertex;
    TIME time;
    HeapNode(VERTEXID _n, TIME _t) : vertex(_n), time(_t) {}
  } HeapNode;
  struct CompareHeapNode {
    bool operator()(const HeapNode &n1, const HeapNode &n2) const {
      return ff(n1.time) > ff(n2.time);
    }
  };
  binomial_heap<HeapNode, compare<CompareHeapNode>> Q;
  vector<binomial_heap<HeapNode, compare<CompareHeapNode>>::handle_type> handles;
  vector<bool> valid_handle;

 public:
  MyHeap() { Q = binomial_heap<HeapNode, compare<CompareHeapNode>>(); }

  MyHeap(VERTEXID num_vertices) {
    Q = binomial_heap<HeapNode, compare<CompareHeapNode>>();
    initialize(num_vertices);
  }

  void initialize(VERTEXID num_vertices) {
    Q.clear();
    handles = vector<binomial_heap<HeapNode, compare<CompareHeapNode>>::handle_type>(num_vertices);
    valid_handle = vector<bool>(num_vertices, false);
  }

  void clear() {
    fill(valid_handle.begin(), valid_handle.end(), false);
    Q.clear();
  }

  inline VERTEXID getMinElement() { return Q.top().vertex; }

  VERTEXID findAndDeleteMinElement() {
    VERTEXID e = Q.top().vertex;
    Q.pop();
    valid_handle[e] = false;
    return e;
  }

  pair<VERTEXID, TIME> getMin() {
    auto heap_element = Q.top();
    return {heap_element.vertex, heap_element.time};
  }

  inline void deleteMin() {
    VERTEXID e = Q.top().vertex;
    Q.pop();
    valid_handle[e] = false;
  }

  void insertElement(VERTEXID v, TIME t) {
    assert(valid_handle[v] == false);
    handles[v] = Q.push(HeapNode(v, t));
    valid_handle[v] = true;
  }

  void adjustHeap(VERTEXID v, TIME t) {
    if (valid_handle[v])
      Q.update(handles[v], HeapNode(v, t));
    else {
      handles[v] = Q.push(HeapNode(v, t));
      valid_handle[v] = true;
    }
  }

  inline bool empty() { return Q.empty(); }

  void print_heap() {
    for (const auto &e : Q) cout << "(" << e.time << ", " << e.vertex << ") | ";
    cout << endl;
  }

  bool is_initialized() { return !valid_handle.empty(); }
};
