#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

//source: https://gist.github.com/goldsborough/f941502e724148b058db6cdf586a0e74

template <typename T>
class BinomialHeap {
 public:
  using size_t = unsigned long;

 private:
  struct Node {
    explicit Node(const T& element_)
    : element(element_), is_invalid(false), parent(this) {
    }

    size_t rank() const noexcept {
      return children.size();
    }

    void add_child(Node* new_child) {
      children.push_back(new_child);
      new_child->parent = this;
    }

    T element;

    bool is_invalid;

    std::vector<Node*> children;
    Node* parent;
  };

 public:
  using ComparisonFunction = std::function<bool(const T&, const T&)>;

  class Handle {
   public:
    const T& element() const noexcept {
      return _node->element;
    }

    bool is_valid() const noexcept {
      return _node->element != nullptr;
    }

    explicit operator bool() const noexcept {
      return is_valid();
    }

   private:
    friend class BinomialHeap;

    explicit Handle(Node* node) : _node(node) {
      assert(node != nullptr);
    }

    Node* _node;
  };

  explicit BinomialHeap(ComparisonFunction greater_priority = std::greater<T>{})
  : _compare(greater_priority), _top(nullptr), _size(0) {
  }

  Handle insert(const T& element) {
    auto node = new Node(element);
    _merge(node);

    if (!_top || _compare(element, _top->element)) {
      _top = node;
    }

    ++_size;

    return Handle(node);
  }

  const T& top() const noexcept {
    assert(!is_empty());

    return _top->element;
  }

  void pop() {
    assert(!is_empty());

    _trees.erase(_trees.find(_top->rank()));

    for (auto& child : _top->children) {
      _merge(child);
    }

    delete _top;
    _top = _find_top();

    --_size;
  }

  void erase(Handle& handle) {
    handle._node->is_invalid = true;
    _swim(handle._node);
    _top = handle._node;

    pop();

    handle._node = nullptr;
  }

  void clear() {
    for (auto& tree : _trees) {
      _clear(tree.second);
    }

    _trees.clear();

    _size = 0;
    _top = nullptr;
  }

  void change(Handle& handle, const T& new_element) {
    const auto& old_element = handle._node->element;
    auto has_greater_priority = _compare(new_element, old_element);

    handle._node->element = new_element;

    if (has_greater_priority) {
      handle._node = _swim(handle._node);
      if (_has_greater_priority(handle._node, _top)) {
        _top = handle._node;
      }

    } else {
      bool was_top = handle._node == _top;
      handle._node = _sink(handle._node);
      if (was_top) _top = _find_top();
    }
  }

  size_t size() const noexcept {
    return _size;
  }

  bool is_empty() const noexcept {
    return size() == 0;
  }

 private:
  using rank_t = unsigned long;

  void _swap(Node* first, Node* second) {
    assert(first != nullptr);
    assert(second != nullptr);

    std::swap(first->element, second->element);
  }

  void _clear(Node* node) {
    assert(node != nullptr);

    for (auto& child : node->children) {
      _clear(child);
    }

    delete node;
  }

  Node* _swim(Node* node) {
    auto parent = node->parent;
    while (_has_greater_priority(node, parent)) {
      _swap(node, parent);
      node = parent;
      parent = node->parent;
    }

    return node;
  }

  Node* _sink(Node* node) {
    while (true) {
      if (node->rank() == 0) break;

      Node* top_child = nullptr;
      for (const auto& child : node->children) {
        if (!top_child || _has_greater_priority(child, top_child)) {
          top_child = child;
        }
      }

      if (!_has_greater_priority(top_child, node)) break;

      _swap(node, top_child);
      node = top_child;
    }

    return node;
  }

  Node* _find_top() const noexcept {
    Node* top = nullptr;
    for (const auto& pair : _trees) {
      auto& tree = pair.second;
      if (!top || _compare(tree->element, top->element)) {
        top = tree;
      }
    }

    return top;
  }

  Node* _combine(Node* first, Node* second) const noexcept {
    if (_has_greater_priority(first, second)) {
      first->add_child(second);
      return first;
    } else {
      second->add_child(first);
      return second;
    }
  }

  void _merge(Node* tree) {
    while (true) {
      auto other = _trees.find(tree->rank());

      if (other == _trees.end()) {
        _trees.emplace(tree->rank(), tree);
        return;
      }

      tree = _combine(tree, other->second);
      _trees.erase(other);
    }
  }

  bool _has_greater_priority(Node* first, Node* second) const
       {
    if (first == second) return false;
    if (first->is_invalid) return true;
    if (second->is_invalid) return false;

    return first->element > second->element;
  }

  ComparisonFunction _compare;
  std::unordered_map<rank_t, Node*> _trees;

  Node* _top;
  size_t _size;
};
