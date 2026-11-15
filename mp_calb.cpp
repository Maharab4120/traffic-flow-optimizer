// ============================================================================
//  MP-CALB : Multi-Path Congestion-Aware Load Balancing for road networks
//  C++17, single file, no dependencies.
//
//  Build :  g++ -O2 -std=c++17 -o mp_calb mp_calb.cpp
//  Run   :  ./mp_calb [--seeds N] [--quick] [--detail] [--generated] [--csv results.csv]
//           --generated  also prints paths / vehicle split / road loads of MP-CALB for
//                        every time step of a GENERATED network (V=20, T=5)
//
//  Problem: connected weighted road graph G=(V,E,w). At each time step a set of
//  requests r_i=(n_i,s_i,d_i) arrives. Split each request's vehicles over one or
//  more s_i->d_i paths so that load is balanced AND total distance is small.
//
//  Proposed method (MP-CALB) minimises the potential
//        Phi(l) = sum_e [ w_e * l_e  +  (lambda/2) * l_e^2 ],
//        lambda = alpha * avg_w / avg_request_size
//  i.e. total travel distance + a convex congestion penalty (a squared-load
//  term, closely tied to load variance). alpha is dimensionless.
//
//    Phase 1  Round-robin platoon assignment: each platoon takes the Dijkstra
//             path under the exact marginal cost  w_e + lambda*(l_e + delta/2).
//    Phase 2  Refinement that strictly decreases Phi (so it must terminate):
//             (a) rip-up-and-reroute of whole sub-flows, (b) unit shifts
//             between a request's existing paths.
//    Guard    Stretch cap: a chosen path longer than `stretch` x shortest is
//             retried with a smaller lambda, then falls back to the shortest
//             path. This defines the "valid path" set implicitly.
//
//  Baselines / ablations (same instances for everyone):
//    SP       static shortest path
//    GLB      greedy load balancing, unsplittable requests, ignores distance
//    GLB-S    GLB but requests split into platoons (isolates the effect of
//             splitting)
//    MPC-1    MP-CALB with no splitting (isolates congestion-awareness)
//    MP-CALB  the proposed method
// ============================================================================
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ============================================================================
// 1. PORTABLE DETERMINISTIC RNG (SplitMix64)
// ============================================================================
class PortableRNG {
    uint64_t state;
public:
    explicit PortableRNG(uint64_t seed = 42ULL) : state(seed) {}
    uint64_t nextUInt64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int nextInt(int low, int high) {
        if (low >= high) return low;
        return low + static_cast<int>(nextUInt64() % static_cast<uint64_t>(high - low + 1));
    }
    double nextDouble(double low, double high) {
        double unit = (nextUInt64() >> 11) * (1.0 / 9007199254740992.0);
        return low + unit * (high - low);
    }
};

// ============================================================================
// 2. DATA STRUCTURES  (classes with public data, per request)
// ============================================================================
class Edge {
public:
    int id = -1;
    int u = 0, v = 0;
    double weight = 0.0;
    Edge() = default;
    Edge(int i, int uu, int vv, double w) : id(i), u(uu), v(vv), weight(w) {}
};

class AdjEntry {
public:
    int to = -1;
    int edge_id = -1;
    double weight = 0.0;
    AdjEntry() = default;
    AdjEntry(int t, int e, double w) : to(t), edge_id(e), weight(w) {}
};

class RoadNetwork {
public:
    int num_vertices = 0;
    vector<Edge> edges;
    vector<vector<AdjEntry>> adj;
    double avg_weight = 1.0;

    explicit RoadNetwork(int n = 0) : num_vertices(n), adj(n) {}

    void addEdge(int u, int v, double w) {
        int id = static_cast<int>(edges.size());
        edges.push_back(Edge(id, u, v, w));
        adj[u].push_back(AdjEntry(v, id, w));
        adj[v].push_back(AdjEntry(u, id, w));
    }
    void finalize() {
        if (edges.empty()) { avg_weight = 1.0; return; }
        double s = 0.0;
        for (const auto& e : edges) s += e.weight;
        avg_weight = s / static_cast<double>(edges.size());
    }
};

class Request {
public:
    int id = 0;
    int vehicles = 0;
    int source = 0, dest = 0;
    Request() = default;
    Request(int i, int v, int s, int d) : id(i), vehicles(v), source(s), dest(d) {}
};

class PathFlow {
public:
    vector<int> vertices;
    vector<int> edge_ids;
    int vehicles = 0;
    double distance = 0.0;
};

class RequestAssignment {
public:
    Request req;
    vector<PathFlow> paths;
};

class StepMetrics {
public:
    double max_load = 0.0, avg_load = 0.0, load_variance = 0.0;
    double total_distance = 0.0, runtime_ms = 0.0;
    double avg_paths = 0.0;
    int fallbacks = 0;
};

class StepResult {
public:
    int time_step = 0;
    vector<RequestAssignment> assignments;
    vector<int> edge_loads;
    StepMetrics metrics;
    double phi_construct = 0.0;
    double phi_final = 0.0;
};

class MPParams {
public:
    double alpha = 0.25;
    int platoons = 6;
    double stretch = 1.5;
    int max_passes = 6;
    bool refine = true;
};

// ============================================================================
// 3. INSTANCE GENERATORS
// ============================================================================
class InstanceGenerator {
public:
    static RoadNetwork createVerificationNetwork() {
        RoadNetwork net(8);
        net.addEdge(0, 1, 4.0); net.addEdge(0, 2, 5.0); net.addEdge(0, 3, 7.0);
        net.addEdge(1, 2, 2.0); net.addEdge(1, 4, 6.0); net.addEdge(2, 3, 3.0);
        net.addEdge(2, 4, 5.0); net.addEdge(2, 5, 6.0); net.addEdge(3, 5, 4.0);
        net.addEdge(4, 5, 3.0); net.addEdge(4, 6, 5.0); net.addEdge(5, 6, 6.0);
        net.addEdge(5, 7, 5.0); net.addEdge(6, 7, 3.0);
        net.finalize();
        return net;
    }

    static vector<vector<Request>> createVerificationRequests() {
        return {
            {Request(1, 60, 0, 7), Request(2, 40, 1, 6), Request(3, 50, 0, 6)},
            {Request(1, 70, 0, 5), Request(2, 45, 2, 7), Request(3, 55, 1, 7), Request(4, 30, 3, 6)},
            {Request(1, 80, 0, 7), Request(2, 65, 1, 5), Request(3, 40, 3, 7)}
        };
    }

    static RoadNetwork generateRandomNetwork(int V, int E, uint64_t seed) {
        PortableRNG rng(seed);
        RoadNetwork net(V);
        int max_possible = V * (V - 1) / 2;
        if (E > max_possible) E = max_possible;
        if (E < V - 1) E = V - 1;

        set<pair<int, int>> have;
        vector<int> perm(V);
        iota(perm.begin(), perm.end(), 0);
        for (int i = V - 1; i > 0; --i) swap(perm[i], perm[rng.nextInt(0, i)]);

        auto weight = [&]() { return round(rng.nextDouble(2.0, 20.0) * 10.0) / 10.0; };
        for (int i = 1; i < V; ++i) {
            int u = perm[i], v = perm[rng.nextInt(0, i - 1)];
            net.addEdge(u, v, weight());
            have.insert({min(u, v), max(u, v)});
        }
        int added = V - 1;
        while (added < E) {
            int u = rng.nextInt(0, V - 1), v = rng.nextInt(0, V - 1);
            if (u == v) continue;
            pair<int, int> key{min(u, v), max(u, v)};
            if (have.count(key)) continue;
            net.addEdge(u, v, weight());
            have.insert(key);
            ++added;
        }
        net.finalize();
        return net;
    }

    static RoadNetwork generateGridNetwork(int V, uint64_t seed, double keep_prob = 0.7) {
        PortableRNG rng(seed);
        int rows = max(2, static_cast<int>(lround(sqrt(static_cast<double>(V)))));
        int cols = max(2, (V + rows - 1) / rows);
        int N = rows * cols;
        RoadNetwork net(N);

        vector<pair<int, int>> cand;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                int id = r * cols + c;
                if (c + 1 < cols) cand.push_back({id, id + 1});
                if (r + 1 < rows) cand.push_back({id, id + cols});
            }
        }
        for (int i = static_cast<int>(cand.size()) - 1; i > 0; --i)
            swap(cand[i], cand[rng.nextInt(0, i)]);

        vector<int> dsu(N);
        iota(dsu.begin(), dsu.end(), 0);
        function<int(int)> find = [&](int x) {
            while (dsu[x] != x) { dsu[x] = dsu[dsu[x]]; x = dsu[x]; }
            return x;
        };
        for (const auto& pr : cand) {
            int ra = find(pr.first), rb = find(pr.second);
            bool keep;
            if (ra != rb) { dsu[ra] = rb; keep = true; }
            else keep = rng.nextDouble(0.0, 1.0) < keep_prob;
            if (keep) net.addEdge(pr.first, pr.second, round(rng.nextDouble(7.0, 13.0) * 10.0) / 10.0);
        }
        net.finalize();
        return net;
    }

    static RoadNetwork generateGeometricNetwork(int V, uint64_t seed, int k = 3) {
        PortableRNG rng(seed);
        RoadNetwork net(V);
        double side = 10.0 * sqrt(static_cast<double>(V));
        vector<double> px(V), py(V);
        for (int i = 0; i < V; ++i) {
            px[i] = rng.nextDouble(0.0, side);
            py[i] = rng.nextDouble(0.0, side);
        }
        auto dist = [&](int a, int b) { return hypot(px[a] - px[b], py[a] - py[b]); };
        auto wOf = [&](int a, int b) { return max(1.0, round(dist(a, b) * 10.0) / 10.0); };

        set<pair<int, int>> have;
        auto tryAdd = [&](int a, int b) {
            if (a == b) return;
            pair<int, int> key{min(a, b), max(a, b)};
            if (have.count(key)) return;
            have.insert(key);
            net.addEdge(a, b, wOf(a, b));
        };

        vector<char> in_tree(V, 0);
        vector<double> key(V, numeric_limits<double>::infinity());
        vector<int> par(V, -1);
        key[0] = 0.0;
        for (int it = 0; it < V; ++it) {
            int u = -1;
            for (int i = 0; i < V; ++i)
                if (!in_tree[i] && (u == -1 || key[i] < key[u])) u = i;
            in_tree[u] = 1;
            if (par[u] != -1) tryAdd(u, par[u]);
            for (int v = 0; v < V; ++v) {
                if (in_tree[v]) continue;
                double d = dist(u, v);
                if (d < key[v]) { key[v] = d; par[v] = u; }
            }
        }
        for (int u = 0; u < V; ++u) {
            vector<pair<double, int>> cand;
            cand.reserve(V - 1);
            for (int v = 0; v < V; ++v)
                if (v != u) cand.push_back({dist(u, v), v});
            int kk = min<int>(k, static_cast<int>(cand.size()));
            partial_sort(cand.begin(), cand.begin() + kk, cand.end());
            for (int j = 0; j < kk; ++j) tryAdd(u, cand[j].second);
        }
        net.finalize();
        return net;
    }

    static vector<vector<Request>> generateRequests(
        int V, int T, int R, uint64_t seed, double hot_frac) {
        PortableRNG rng(seed);
        int H = min(V, max(3, V / 40));
        vector<int> hubs;
        set<int> used;
        while (static_cast<int>(hubs.size()) < H) {
            int h = rng.nextInt(0, V - 1);
            if (used.insert(h).second) hubs.push_back(h);
        }
        vector<vector<Request>> steps(T);
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < R; ++r) {
                int s, d;
                if (H >= 2 && rng.nextDouble(0.0, 1.0) < hot_frac) {
                    s = hubs[rng.nextInt(0, H - 1)];
                    d = hubs[rng.nextInt(0, H - 1)];
                    while (d == s) d = hubs[rng.nextInt(0, H - 1)];
                } else {
                    s = rng.nextInt(0, V - 1);
                    d = rng.nextInt(0, V - 1);
                    while (d == s) d = rng.nextInt(0, V - 1);
                }
                steps[t].push_back(Request(r + 1, rng.nextInt(20, 100), s, d));
            }
        }
        return steps;
    }
};

// ============================================================================
// 4. ROUTING ENGINE
// ============================================================================
class TrafficRouter {
public:
    template <typename CostFunc>
    static PathFlow dijkstraPath(const RoadNetwork& net, int src, int dst, CostFunc cost_fn) {
        const double INF = numeric_limits<double>::infinity();
        int V = net.num_vertices;
        vector<double> dist(V, INF);
        vector<int> parent_v(V, -1), parent_e(V, -1);
        using Node = pair<double, int>;
        priority_queue<Node, vector<Node>, greater<Node>> pq;

        dist[src] = 0.0;
        pq.push({0.0, src});
        while (!pq.empty()) {
            auto [d, u] = pq.top();
            pq.pop();
            if (d > dist[u] + 1e-12) continue;
            if (u == dst) break;
            for (const auto& nb : net.adj[u]) {
                double nd = d + cost_fn(nb.edge_id, nb.weight);
                if (nd + 1e-12 < dist[nb.to] ||
                    (abs(nd - dist[nb.to]) <= 1e-12 && u < parent_v[nb.to])) {
                    dist[nb.to] = nd;
                    parent_v[nb.to] = u;
                    parent_e[nb.to] = nb.edge_id;
                    pq.push({nd, nb.to});
                }
            }
        }
        PathFlow pf;
        for (int cur = dst; cur != -1; cur = parent_v[cur]) {
            pf.vertices.push_back(cur);
            int eid = parent_e[cur];
            if (eid != -1) {
                pf.edge_ids.push_back(eid);
                pf.distance += net.edges[eid].weight;
            }
        }
        reverse(pf.vertices.begin(), pf.vertices.end());
        reverse(pf.edge_ids.begin(), pf.edge_ids.end());
        return pf;
    }

    static PathFlow shortestPath(const RoadNetwork& net, int s, int d) {
        return dijkstraPath(net, s, d, [](int, double w) { return w; });
    }

    static void addFlow(vector<PathFlow>& paths, const PathFlow& pf, int delta) {
        for (auto& p : paths) {
            if (p.vertices == pf.vertices) { p.vehicles += delta; return; }
        }
        PathFlow np = pf;
        np.vehicles = delta;
        paths.push_back(np);
    }

    static void finalizeAssignments(vector<RequestAssignment>& asg) {
        for (auto& a : asg) {
            auto& ps = a.paths;
            ps.erase(remove_if(ps.begin(), ps.end(),
                     [](const PathFlow& p) { return p.vehicles <= 0; }), ps.end());
            stable_sort(ps.begin(), ps.end(),
                [](const PathFlow& x, const PathFlow& y) { return x.vehicles > y.vehicles; });
        }
    }

    static StepMetrics computeMetrics(const RoadNetwork& net, const vector<int>& loads,
                                      double runtime_ms,
                                      const vector<RequestAssignment>& asg, int fallbacks) {
        StepMetrics m;
        m.runtime_ms = runtime_ms;
        m.fallbacks = fallbacks;
        int M = static_cast<int>(net.edges.size());
        if (M == 0) return m;
        double sum = 0.0, mx = 0.0, dist = 0.0;
        for (int e = 0; e < M; ++e) {
            double l = loads[e];
            sum += l;
            mx = max(mx, l);
            dist += l * net.edges[e].weight;
        }
        double avg = sum / M, var = 0.0;
        for (int e = 0; e < M; ++e) var += (loads[e] - avg) * (loads[e] - avg);
        m.max_load = mx;
        m.avg_load = avg;
        m.load_variance = var / M;
        m.total_distance = dist;
        if (!asg.empty()) {
            double np = 0.0;
            for (const auto& a : asg) np += static_cast<double>(a.paths.size());
            m.avg_paths = np / static_cast<double>(asg.size());
        }
        return m;
    }

    static double seconds(chrono::high_resolution_clock::time_point a,
                          chrono::high_resolution_clock::time_point b) {
        return chrono::duration<double, milli>(b - a).count();
    }

    // ---- Baseline 1: shortest path ----------------------------------------
    static StepResult solveShortestPath(const RoadNetwork& net,
                                        const vector<Request>& requests, int t) {
        auto t0 = chrono::high_resolution_clock::now();
        StepResult res;
        res.time_step = t;
        res.edge_loads.assign(net.edges.size(), 0);
        map<pair<int, int>, PathFlow> cache;
        for (const auto& req : requests) {
            auto key = make_pair(req.source, req.dest);
            auto it = cache.find(key);
            if (it == cache.end()) it = cache.emplace(key, shortestPath(net, req.source, req.dest)).first;
            PathFlow pf = it->second;
            pf.vehicles = req.vehicles;
            for (int e : pf.edge_ids) res.edge_loads[e] += req.vehicles;
            RequestAssignment ra;
            ra.req = req;
            ra.paths.push_back(pf);
            res.assignments.push_back(ra);
        }
        auto t1 = chrono::high_resolution_clock::now();
        res.metrics = computeMetrics(net, res.edge_loads, seconds(t0, t1), res.assignments, 0);
        return res;
    }

    // ---- Baseline 2: greedy load balancing --------------------------------
    static StepResult solveGreedyLB(const RoadNetwork& net, const vector<Request>& requests,
                                    int t, int platoons) {
        auto t0 = chrono::high_resolution_clock::now();
        StepResult res;
        res.time_step = t;
        res.edge_loads.assign(net.edges.size(), 0);
        int R = static_cast<int>(requests.size());
        res.assignments.resize(R);
        for (int i = 0; i < R; ++i) res.assignments[i].req = requests[i];

        vector<int> order(R);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            if (requests[a].vehicles != requests[b].vehicles)
                return requests[a].vehicles > requests[b].vehicles;
            return requests[a].id < requests[b].id;
        });

        vector<int> remaining(R), chunk(R);
        for (int i = 0; i < R; ++i) {
            remaining[i] = requests[i].vehicles;
            chunk[i] = max(1, (requests[i].vehicles + platoons - 1) / platoons);
        }
        bool any = true;
        while (any) {
            any = false;
            for (int idx : order) {
                if (remaining[idx] <= 0) continue;
                any = true;
                int delta = min(remaining[idx], chunk[idx]);
                remaining[idx] -= delta;
                PathFlow pf = dijkstraPath(net, requests[idx].source, requests[idx].dest,
                    [&](int e, double w) {
                        double cur = res.edge_loads[e], nx = cur + delta;
                        return (nx * nx - cur * cur) + 0.05 * w;
                    });
                for (int e : pf.edge_ids) res.edge_loads[e] += delta;
                addFlow(res.assignments[idx].paths, pf, delta);
            }
        }
        finalizeAssignments(res.assignments);
        auto t1 = chrono::high_resolution_clock::now();
        res.metrics = computeMetrics(net, res.edge_loads, seconds(t0, t1), res.assignments, 0);
        return res;
    }

    // ---- Proposed: MP-CALB -------------------------------------------------
    static StepResult solveMPCALB(const RoadNetwork& net, const vector<Request>& requests,
                                  int t, const MPParams& P) {
        auto t0 = chrono::high_resolution_clock::now();
        const int R = static_cast<int>(requests.size());
        const int M = static_cast<int>(net.edges.size());

        StepResult res;
        res.time_step = t;
        res.edge_loads.assign(M, 0);
        res.assignments.resize(R);
        for (int i = 0; i < R; ++i) res.assignments[i].req = requests[i];

        double n_bar = 0.0;
        for (const auto& r : requests) n_bar += r.vehicles;
        n_bar = R > 0 ? n_bar / R : 1.0;
        const double lambda = P.alpha * net.avg_weight / max(1.0, n_bar);

        map<pair<int, int>, double> d0cache;
        vector<double> d0(R);
        for (int i = 0; i < R; ++i) {
            auto key = make_pair(requests[i].source, requests[i].dest);
            auto it = d0cache.find(key);
            if (it == d0cache.end())
                it = d0cache.emplace(key, shortestPath(net, key.first, key.second).distance).first;
            d0[i] = it->second;
        }

        vector<int> order(R);
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            double wa = requests[a].vehicles * d0[a], wb = requests[b].vehicles * d0[b];
            if (wa != wb) return wa > wb;
            return requests[a].id < requests[b].id;
        });

        int fallbacks = 0;

        auto routeGuarded = [&](int src, int dst, int delta, double base, bool count) {
            double scale = 1.0;
            for (int attempt = 0; attempt < 3; ++attempt) {
                double lam = lambda * scale;
                PathFlow pf = dijkstraPath(net, src, dst, [&](int e, double w) {
                    return w + lam * (res.edge_loads[e] + 0.5 * delta);
                });
                if (pf.distance <= P.stretch * base + 1e-9) return pf;
                scale *= 0.5;
            }
            if (count) ++fallbacks;
            return shortestPath(net, src, dst);
        };

        auto phiTotal = [&]() {
            double s = 0.0;
            for (int e = 0; e < M; ++e) {
                double l = res.edge_loads[e];
                s += net.edges[e].weight * l + 0.5 * lambda * l * l;
            }
            return s;
        };

        // Phase 1
        vector<int> remaining(R), chunk(R);
        for (int i = 0; i < R; ++i) {
            remaining[i] = requests[i].vehicles;
            chunk[i] = max(1, (requests[i].vehicles + P.platoons - 1) / P.platoons);
        }
        bool any = true;
        while (any) {
            any = false;
            for (int i : order) {
                if (remaining[i] <= 0) continue;
                any = true;
                int delta = min(remaining[i], chunk[i]);
                remaining[i] -= delta;
                PathFlow pf = routeGuarded(requests[i].source, requests[i].dest, delta, d0[i], true);
                for (int e : pf.edge_ids) res.edge_loads[e] += delta;
                addFlow(res.assignments[i].paths, pf, delta);
            }
        }
        res.phi_construct = phiTotal();

        // Phase 2: refinement
        if (P.refine && lambda > 0.0) {
            vector<int> st_old(M, 0), st_new(M, 0), st_seen(M, 0);
            int stamp = 0;

            auto tryReroute = [&](int i, size_t a) -> bool {
                auto& asg = res.assignments[i];
                int delta = asg.paths[a].vehicles;
                if (delta <= 0) return false;
                const vector<int> old_edges = asg.paths[a].edge_ids;
                const vector<int> old_vertices = asg.paths[a].vertices;
                for (int e : old_edges) res.edge_loads[e] -= delta;
                PathFlow np = routeGuarded(asg.req.source, asg.req.dest, delta, d0[i], false);
                if (np.vertices == old_vertices) {
                    for (int e : old_edges) res.edge_loads[e] += delta;
                    return false;
                }
                ++stamp;
                for (int e : old_edges) st_old[e] = stamp;
                for (int e : np.edge_ids) st_new[e] = stamp;
                double dphi = 0.0;
                auto visit = [&](int e) {
                    if (st_seen[e] == stamp) return;
                    st_seen[e] = stamp;
                    double w = net.edges[e].weight, base_l = res.edge_loads[e];
                    double l_old = base_l + (st_old[e] == stamp ? delta : 0);
                    double l_new = base_l + (st_new[e] == stamp ? delta : 0);
                    dphi += (w * l_new + 0.5 * lambda * l_new * l_new)
                          - (w * l_old + 0.5 * lambda * l_old * l_old);
                };
                for (int e : old_edges) visit(e);
                for (int e : np.edge_ids) visit(e);
                if (dphi < -1e-9) {
                    for (int e : np.edge_ids) res.edge_loads[e] += delta;
                    asg.paths[a].vehicles = 0;
                    addFlow(asg.paths, np, delta);
                    return true;
                }
                for (int e : old_edges) res.edge_loads[e] += delta;
                return false;
            };

            auto unitShifts = [&](int i) -> bool {
                auto& paths = res.assignments[i].paths;
                bool improved = false;
                for (size_t a = 0; a < paths.size(); ++a) {
                    for (size_t b = 0; b < paths.size(); ++b) {
                        if (a == b) continue;
                        while (paths[a].vehicles > 0) {
                            double d = 0.0;
                            for (int e : paths[a].edge_ids)
                                d -= net.edges[e].weight + lambda * (res.edge_loads[e] - 0.5);
                            for (int e : paths[a].edge_ids) res.edge_loads[e] -= 1;
                            for (int e : paths[b].edge_ids)
                                d += net.edges[e].weight + lambda * (res.edge_loads[e] + 0.5);
                            if (d < -1e-9) {
                                for (int e : paths[b].edge_ids) res.edge_loads[e] += 1;
                                paths[a].vehicles -= 1;
                                paths[b].vehicles += 1;
                                improved = true;
                            } else {
                                for (int e : paths[a].edge_ids) res.edge_loads[e] += 1;
                                break;
                            }
                        }
                    }
                }
                return improved;
            };

            for (int pass = 0; pass < P.max_passes; ++pass) {
                bool improved = false;
                for (int i : order) {
                    for (size_t a = 0; a < res.assignments[i].paths.size(); ++a)
                        if (tryReroute(i, a)) improved = true;
                    if (unitShifts(i)) improved = true;
                }
                if (!improved) break;
            }
        }
        res.phi_final = phiTotal();

        finalizeAssignments(res.assignments);
        auto t1 = chrono::high_resolution_clock::now();
        res.metrics = computeMetrics(net, res.edge_loads, seconds(t0, t1), res.assignments, fallbacks);
        return res;
    }
};

// ============================================================================
// 5. ALGORITHM REGISTRY, STATISTICS, FORMATTING
// ============================================================================
using SolveFn = function<StepResult(const RoadNetwork&, const vector<Request>&, int)>;

class AlgoSpec {
public:
    string name;
    SolveFn run;
    AlgoSpec() = default;
    AlgoSpec(const string& n, SolveFn f) : name(n), run(std::move(f)) {}
};

static vector<AlgoSpec> standardAlgorithms(const MPParams& P) {
    vector<AlgoSpec> v;
    v.push_back({"SP", [](const RoadNetwork& n, const vector<Request>& r, int t) {
        return TrafficRouter::solveShortestPath(n, r, t); }});
    v.push_back({"GLB", [](const RoadNetwork& n, const vector<Request>& r, int t) {
        return TrafficRouter::solveGreedyLB(n, r, t, 1); }});
    v.push_back({"GLB-S", [P](const RoadNetwork& n, const vector<Request>& r, int t) {
        return TrafficRouter::solveGreedyLB(n, r, t, P.platoons); }});
    v.push_back({"MPC-1", [P](const RoadNetwork& n, const vector<Request>& r, int t) {
        MPParams q = P; q.platoons = 1;
        return TrafficRouter::solveMPCALB(n, r, t, q); }});
    v.push_back({"MP-CALB", [P](const RoadNetwork& n, const vector<Request>& r, int t) {
        return TrafficRouter::solveMPCALB(n, r, t, P); }});
    return v;
}

class Stat {
public:
    int n = 0;
    double mean = 0.0, m2 = 0.0;
    void add(double x) {
        ++n;
        double d = x - mean;
        mean += d / n;
        m2 += d * (x - mean);
    }
    double sd() const { return n > 1 ? sqrt(m2 / (n - 1)) : 0.0; }
};

class Agg {
public:
    Stat max_load, avg_load, var, dist, time, stretch, max_ratio, paths, fb;
    void add(const StepMetrics& m, const StepMetrics& sp) {
        max_load.add(m.max_load);
        avg_load.add(m.avg_load);
        var.add(m.load_variance);
        dist.add(m.total_distance);
        time.add(m.runtime_ms);
        stretch.add(sp.total_distance > 0 ? m.total_distance / sp.total_distance : 1.0);
        max_ratio.add(sp.max_load > 0 ? m.max_load / sp.max_load : 1.0);
        paths.add(m.avg_paths);
        fb.add(m.fallbacks);
    }
};

static string fmt(double v, int prec) {
    ostringstream o;
    o << fixed << setprecision(prec) << v;
    return o.str();
}
static string fmtPM(const Stat& s, int prec) { return fmt(s.mean, prec) + "+-" + fmt(s.sd(), prec); }

static string formatPath(const vector<int>& verts) {
    ostringstream oss;
    for (size_t i = 0; i < verts.size(); ++i) {
        oss << verts[i];
        if (i + 1 < verts.size()) oss << "->";
    }
    return oss.str();
}

static void printDetailedStepReport(const RoadNetwork& net, const StepResult& res,
                                    const string& alg) {
    cout << "\n----------------------------------------------------------------------------\n";
    cout << " TIME STEP t = " << res.time_step << " | PATHS, VEHICLE DISTRIBUTION, ROAD LOADS | " << alg << "\n";
    cout << "----------------------------------------------------------------------------\n";
    for (const auto& a : res.assignments) {
        cout << " Request r" << a.req.id << ": " << a.req.vehicles << " vehicles, node "
             << a.req.source << " -> node " << a.req.dest
             << " (" << a.paths.size() << " path(s))\n";
        for (size_t p = 0; p < a.paths.size(); ++p) {
            const auto& pf = a.paths[p];
            double pct = 100.0 * pf.vehicles / a.req.vehicles;
            cout << "   [Path " << (p + 1) << "] " << setw(18) << left << formatPath(pf.vertices)
                 << " | Vehicles: " << setw(3) << right << pf.vehicles
                 << " (" << fixed << setprecision(1) << setw(5) << pct << "%)"
                 << " | Path dist: " << pf.distance << "\n";
        }
    }
    cout << "\n Road-wise loads at t = " << res.time_step << ":\n";
    cout << "   Edge | Road (u <-> v) | Weight | Load\n";
    cout << "  ------+----------------+--------+------\n";
    for (const auto& e : net.edges) {
        cout << "   e" << setw(3) << left << e.id << " |  "
             << setw(2) << right << e.u << " <-> " << setw(2) << left << e.v
             << "     | " << setw(6) << right << fixed << setprecision(1) << e.weight
             << " | " << setw(4) << res.edge_loads[e.id] << "\n";
    }
    const auto& m = res.metrics;
    cout << "  Metrics -> Max: " << setprecision(1) << m.max_load
         << " | Avg: " << setprecision(2) << m.avg_load
         << " | Var: " << m.load_variance
         << " | Total dist: " << setprecision(1) << m.total_distance << "\n";
    if (res.phi_final > 0.0)
        cout << "  Potential Phi: after Phase 1 = " << setprecision(2) << res.phi_construct
             << " -> after Phase 2 = " << res.phi_final << "\n";
}

// ============================================================================
// 6. EXPERIMENTS
// ============================================================================
enum class Family { GRID, GEOMETRIC, RANDOM };

class ScaleConfig {
public:
    string label;
    Family family = Family::GRID;
    int V = 0;
    int E_target = 0;
    int T = 0;
    int R = 0;
    uint64_t seed = 0;

    ScaleConfig() = default;
    ScaleConfig(const string& l, Family f, int v, int e, int t, int r, uint64_t s)
        : label(l), family(f), V(v), E_target(e), T(t), R(r), seed(s) {}
};

static RoadNetwork makeNetwork(const ScaleConfig& c, uint64_t seed) {
    switch (c.family) {
        case Family::GRID:      return InstanceGenerator::generateGridNetwork(c.V, seed);
        case Family::GEOMETRIC: return InstanceGenerator::generateGeometricNetwork(c.V, seed);
        default:                return InstanceGenerator::generateRandomNetwork(c.V, c.E_target, seed);
    }
}

static const double HOT_FRAC = 0.5;

// Independent consistency check of a StepResult.
static bool validateResult(const RoadNetwork& net, const StepResult& r, double stretch_cap = -1.0) {
    vector<long long> loads(net.edges.size(), 0);
    map<pair<int, int>, double> d0cache;
    for (const auto& a : r.assignments) {
        long long sum = 0;
        double d0 = 0.0;
        if (stretch_cap > 0.0) {
            auto key = make_pair(a.req.source, a.req.dest);
            auto it = d0cache.find(key);
            if (it == d0cache.end())
                it = d0cache.emplace(key, TrafficRouter::shortestPath(net, key.first, key.second).distance).first;
            d0 = it->second;
        }
        for (const auto& p : a.paths) {
            if (p.vehicles < 0 || p.vertices.empty()) return false;
            if (p.vertices.size() < 2 && a.req.source != a.req.dest) return false;
            if (p.vertices.front() != a.req.source || p.vertices.back() != a.req.dest) return false;
            if (p.edge_ids.size() + 1 != p.vertices.size()) return false;
            // simple path: no vertex repeated
            set<int> seen(p.vertices.begin(), p.vertices.end());
            if (seen.size() != p.vertices.size()) return false;
            // valid path: not longer than stretch_cap x shortest (only when a cap is promised)
            if (stretch_cap > 0.0 && p.distance > stretch_cap * d0 + 1e-9) return false;
            for (size_t i = 0; i < p.edge_ids.size(); ++i) {
                const Edge& e = net.edges[p.edge_ids[i]];
                int x = p.vertices[i], y = p.vertices[i + 1];
                if (!((e.u == x && e.v == y) || (e.u == y && e.v == x))) return false;
                loads[e.id] += p.vehicles;
            }
            sum += p.vehicles;
        }
        if (sum != a.req.vehicles) return false;
    }
    for (size_t e = 0; e < loads.size(); ++e)
        if (loads[e] != r.edge_loads[e]) return false;
    return true;
}

// Only the MP-CALB family promises the stretch cap; SP / GLB paths are not capped.
static double capFor(const string& name, const MPParams& P) {
    return name.rfind("MP", 0) == 0 ? P.stretch : -1.0;
}

static void runBenchmark(const vector<ScaleConfig>& scales, const MPParams& P,
                         int num_seeds, ofstream* csv) {
    auto algos = standardAlgorithms(P);
    long long checked = 0, invalid = 0;
    cout << "\n============================================================================\n";
    cout << "  MULTI-SCALE BENCHMARK  (" << num_seeds << " seeds x T steps, mean +- std)\n";
    cout << "============================================================================\n";
    cout << " Alg      | Max load       | Avg load | Load var     | Total dist       | Stretch | Paths/req | ms\n";
    cout << "----------+----------------+----------+--------------+------------------+---------+-----------+--------\n";
    for (const auto& cfg : scales) {
        vector<Agg> agg(algos.size());
        int E_seen = 0, V_seen = 0;
        for (int s = 0; s < num_seeds; ++s) {
            RoadNetwork net = makeNetwork(cfg, cfg.seed + 7919ULL * s);
            if (s == 0) { E_seen = static_cast<int>(net.edges.size()); V_seen = net.num_vertices; }
            auto reqs = InstanceGenerator::generateRequests(
                net.num_vertices, cfg.T, cfg.R, cfg.seed + 555ULL + 104729ULL * s, HOT_FRAC);
            for (int t = 0; t < cfg.T; ++t) {
                vector<StepResult> rs;
                for (auto& a : algos) rs.push_back(a.run(net, reqs[t], t + 1));
                for (size_t k = 0; k < algos.size(); ++k) {
                    ++checked;
                    if (!validateResult(net, rs[k], capFor(algos[k].name, P))) ++invalid;
                    agg[k].add(rs[k].metrics, rs[0].metrics);
                }
            }
        }
        cout << " " << cfg.label << "  [V=" << V_seen << ", E=" << E_seen
             << ", R=" << cfg.R << "/step, T=" << cfg.T << "]\n";
        for (size_t k = 0; k < algos.size(); ++k) {
            const Agg& a = agg[k];
            cout << " " << setw(8) << left << algos[k].name
                 << " | " << setw(14) << fmtPM(a.max_load, 1)
                 << " | " << setw(8) << fmt(a.avg_load.mean, 1)
                 << " | " << setw(12) << fmt(a.var.mean, 0)
                 << " | " << setw(16) << fmtPM(a.dist, 0)
                 << " | " << setw(7) << fmt(a.stretch.mean, 3)
                 << " | " << setw(9) << fmt(a.paths.mean, 2)
                 << " | " << fmt(a.time.mean, 2) << "\n";
            if (csv) {
                *csv << cfg.label << "," << algos[k].name << ","
                     << a.max_load.mean << "," << a.max_load.sd() << ","
                     << a.avg_load.mean << "," << a.var.mean << "," << a.var.sd() << ","
                     << a.dist.mean << "," << a.dist.sd() << ","
                     << a.stretch.mean << "," << a.time.mean << "," << a.paths.mean << "\n";
            }
        }
        cout << "----------+----------------+----------+--------------+------------------+---------+-----------+--------\n";
    }
    cout << " Validation: " << (checked - invalid) << "/" << checked
         << " results passed (vehicle conservation, simple valid paths, stretch cap, load consistency)\n";
    if (invalid > 0) {
        cerr << "!!! WARNING: " << invalid << " invalid results detected !!!\n";
    }
}

static void runSweep(const string& title,
                     const vector<pair<string, MPParams>>& configs,
                     const ScaleConfig& cfg, int num_seeds) {
    cout << "\n============================================================================\n";
    cout << "  " << title << "\n  (" << cfg.label << ", " << num_seeds << " seeds x " << cfg.T << " steps)\n";
    cout << "============================================================================\n";
    cout << " Setting  | Max load | Load var | Total dist | Dist vs SP | Max vs SP | Paths/req | Fallbk\n";
    cout << "----------+----------+----------+------------+------------+-----------+-----------+-------\n";

    vector<string> names{"SP"};
    vector<Agg> agg(configs.size() + 1);
    for (const auto& c : configs) names.push_back(c.first);

    for (int s = 0; s < num_seeds; ++s) {
        RoadNetwork net = makeNetwork(cfg, cfg.seed + 7919ULL * s);
        auto reqs = InstanceGenerator::generateRequests(
            net.num_vertices, cfg.T, cfg.R, cfg.seed + 555ULL + 104729ULL * s, HOT_FRAC);
        for (int t = 0; t < cfg.T; ++t) {
            StepResult sp = TrafficRouter::solveShortestPath(net, reqs[t], t + 1);
            agg[0].add(sp.metrics, sp.metrics);
            for (size_t k = 0; k < configs.size(); ++k) {
                StepResult r = TrafficRouter::solveMPCALB(net, reqs[t], t + 1, configs[k].second);
                agg[k + 1].add(r.metrics, sp.metrics);
            }
        }
    }
    for (size_t k = 0; k < agg.size(); ++k) {
        const Agg& a = agg[k];
        cout << " " << setw(8) << left << names[k]
             << " | " << setw(8) << right << fmt(a.max_load.mean, 1)
             << " | " << setw(8) << fmt(a.var.mean, 0)
             << " | " << setw(10) << fmt(a.dist.mean, 0)
             << " | " << setw(9) << fmt((a.stretch.mean - 1.0) * 100.0, 1) << "%"
             << " | " << setw(8) << fmt((a.max_ratio.mean - 1.0) * 100.0, 1) << "%"
             << " | " << setw(9) << fmt(a.paths.mean, 2)
             << " | " << setw(5) << fmt(a.fb.mean, 1) << "\n";
    }
    cout << "----------+----------+----------+------------+------------+-----------+-----------+-------\n";
}

static void runScalability(const MPParams& P) {
    cout << "\n============================================================================\n";
    cout << "  SCALABILITY (perturbed grids, R = V/5 requests/step, mean of 2 steps)\n";
    cout << "============================================================================\n";
    cout << "   V   |   E   |  R  | SP (ms) | MP-CALB (ms) | log-log slope of MP-CALB time\n";
    cout << "-------+-------+-----+---------+--------------+------------------------------\n";
    vector<int> sizes = {100, 200, 400, 800, 1600};
    double prev_t = 0.0;
    int prev_V = 0;
    for (int V : sizes) {
        RoadNetwork net = InstanceGenerator::generateGridNetwork(V, 9000ULL + V);
        int NV = net.num_vertices;
        int R = max(10, NV / 5);
        auto reqs = InstanceGenerator::generateRequests(NV, 2, R, 777ULL + V, HOT_FRAC);
        double t_sp = 0.0, t_mp = 0.0;
        for (int t = 0; t < 2; ++t) {
            t_sp += TrafficRouter::solveShortestPath(net, reqs[t], t + 1).metrics.runtime_ms / 2.0;
            t_mp += TrafficRouter::solveMPCALB(net, reqs[t], t + 1, P).metrics.runtime_ms / 2.0;
        }
        string slope = "-";
        if (prev_V > 0 && prev_t > 0 && t_mp > 0)
            slope = fmt(log(t_mp / prev_t) / log(static_cast<double>(NV) / prev_V), 2);
        cout << " " << setw(5) << right << NV << " | " << setw(5) << net.edges.size()
             << " | " << setw(3) << R << " | " << setw(7) << fmt(t_sp, 2)
             << " | " << setw(12) << fmt(t_mp, 2) << " | " << slope << "\n";
        prev_t = t_mp;
        prev_V = NV;
    }
    cout << "-------+-------+-----+---------+--------------+------------------------------\n";
}

// ============================================================================
// 7. MAIN
// ============================================================================
// ============================================================================
// 7. OPTIMALITY CHECK  (how close is MP-CALB to the best possible Phi?)
// ============================================================================
// Frank-Wolfe on the continuous relaxation of Phi (fractional flows, no integer
// vehicles, no stretch cap). Phi is convex, so for any load vector l and the
// all-or-nothing flow y that routes every request on its shortest path under the
// gradient costs g_e = w_e + lambda*l_e:
//        Phi*  >=  Phi(l) + g . (y - l)            (a valid LOWER BOUND)
// Every feasible routing (including MP-CALB's) therefore has Phi >= this bound.
static double phiOfLoads(const RoadNetwork& net, const vector<double>& l, double lambda) {
    double s = 0.0;
    for (size_t e = 0; e < l.size(); ++e) s += net.edges[e].weight * l[e] + 0.5 * lambda * l[e] * l[e];
    return s;
}

static double frankWolfeLowerBound(const RoadNetwork& net, const vector<Request>& reqs,
                                   double lambda, int iters) {
    const int M = static_cast<int>(net.edges.size());
    vector<double> l(M, 0.0);
    for (const auto& r : reqs) {
        PathFlow p = TrafficRouter::shortestPath(net, r.source, r.dest);
        for (int e : p.edge_ids) l[e] += r.vehicles;
    }
    double best = -numeric_limits<double>::infinity();
    for (int k = 0; k < iters; ++k) {
        vector<double> g(M), y(M, 0.0);
        for (int e = 0; e < M; ++e) g[e] = net.edges[e].weight + lambda * l[e];
        for (const auto& r : reqs) {
            PathFlow p = TrafficRouter::dijkstraPath(net, r.source, r.dest,
                                                     [&](int e, double) { return g[e]; });
            for (int e : p.edge_ids) y[e] += r.vehicles;
        }
        double gd = 0.0, dd = 0.0;
        for (int e = 0; e < M; ++e) {
            double d = y[e] - l[e];
            gd += g[e] * d;
            dd += d * d;
        }
        best = max(best, phiOfLoads(net, l, lambda) + gd);
        if (dd < 1e-12 || lambda <= 0.0) break;
        double gamma = min(1.0, max(0.0, -gd / (lambda * dd)));   // exact line search
        for (int e = 0; e < M; ++e) l[e] += gamma * (y[e] - l[e]);
    }
    return best;
}

static void runOptimalityCheck(const vector<ScaleConfig>& scales, const MPParams& P, int num_seeds) {
    auto algos = standardAlgorithms(P);   // order: SP, GLB, GLB-S, MPC-1, MP-CALB
    cout << "\n============================================================================\n";
    cout << "  OPTIMALITY CHECK: Phi of each method vs a Frank-Wolfe lower bound\n";
    cout << "  (Phi = distance + lambda/2 * sum of squared loads, same lambda for all)\n";
    cout << "============================================================================\n";
    cout << " Network                 | Phi SP   | Phi GLB  | Phi MP-CALB | Lower bound | MP-CALB gap | no-cap gap | GLB gap | SP->LB gain reached\n";
    cout << "-------------------------+----------+----------+-------------+-------------+-------------+------------+---------+--------------------\n";
    for (const auto& cfg : scales) {
        if (cfg.V > 600) continue;   // keep the check fast: skip the X-Large network
        double sSP = 0, sGL = 0, sMP = 0, sLB = 0, sNC = 0, gainMP = 0;
        int n = 0, ng = 0;
        for (int sd = 0; sd < num_seeds; ++sd) {
            RoadNetwork net = makeNetwork(cfg, cfg.seed + 7919ULL * sd);
            auto reqs = InstanceGenerator::generateRequests(
                net.num_vertices, cfg.T, cfg.R, cfg.seed + 555ULL + 104729ULL * sd, HOT_FRAC);
            for (int t = 0; t < cfg.T; ++t) {
                double nb = 0.0;
                for (const auto& r : reqs[t]) nb += r.vehicles;
                nb /= static_cast<double>(reqs[t].size());
                double lambda = P.alpha * net.avg_weight / max(1.0, nb);
                auto toD = [](const vector<int>& v) { return vector<double>(v.begin(), v.end()); };
                double pSP = phiOfLoads(net, toD(algos[0].run(net, reqs[t], t + 1).edge_loads), lambda);
                double pGL = phiOfLoads(net, toD(algos[1].run(net, reqs[t], t + 1).edge_loads), lambda);
                double pMP = phiOfLoads(net, toD(algos[4].run(net, reqs[t], t + 1).edge_loads), lambda);
                MPParams nocap = P;
                nocap.stretch = 1e9;   // same algorithm, stretch cap removed
                double pNC = phiOfLoads(net, toD(TrafficRouter::solveMPCALB(net, reqs[t], t + 1, nocap).edge_loads), lambda);
                double lb = frankWolfeLowerBound(net, reqs[t], lambda, 300);
                sNC += pNC;
                sSP += pSP; sGL += pGL; sMP += pMP; sLB += lb; ++n;
                if (pSP - lb > 1e-6) { gainMP += (pSP - pMP) / (pSP - lb); ++ng; }
            }
        }
        cout << " " << setw(23) << left << cfg.label << " | " << setw(8) << right << fmt(sSP / n, 0)
             << " | " << setw(8) << fmt(sGL / n, 0)
             << " | " << setw(11) << fmt(sMP / n, 0)
             << " | " << setw(11) << fmt(sLB / n, 0)
             << " | " << setw(10) << fmt((sMP / sLB - 1.0) * 100.0, 2) << "% "
             << " | " << setw(9) << fmt((sNC / sLB - 1.0) * 100.0, 2) << "% "
             << " | " << setw(6) << fmt((sGL / sLB - 1.0) * 100.0, 2) << "% "
             << " | " << setw(10) << fmt(ng ? gainMP / ng * 100.0 : 100.0, 1) << "%\n";
    }
    cout << "-------------------------+----------+----------+-------------+-------------+-------------+------------+---------+--------------------\n";
    cout << " The bound ignores integrality and the stretch cap, so the true optimum lies between the bound\n";
    cout << " and MP-CALB. no-cap gap = the same algorithm with the stretch cap removed: what is left of the\n";
    cout << " gap is the price of the cap, not a weakness of the search.\n";
}

int main(int argc, char** argv) {
    int num_seeds = 5;
    bool quick = false, detail = false, generated = false;
    string csv_path;
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--seeds" && i + 1 < argc) num_seeds = max(1, atoi(argv[++i]));
        else if (a == "--quick") quick = true;
        else if (a == "--detail") detail = true;
        else if (a == "--generated") generated = true;
        else if (a == "--csv" && i + 1 < argc) csv_path = argv[++i];
    }
    ofstream csv_file;
    ofstream* csv = nullptr;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        csv_file << "scale,algorithm,max_load_mean,max_load_std,avg_load_mean,var_mean,var_std,"
                    "dist_mean,dist_std,stretch_mean,time_ms_mean,paths_per_req_mean\n";
        csv = &csv_file;
    }

    MPParams P;
    cout << "============================================================================\n";
    cout << "  MP-CALB: TRAFFIC LOAD BALANCING + DISTANCE MINIMISATION (C++17)\n";
    cout << "  alpha=" << P.alpha << "  platoons=" << P.platoons << "  stretch<=" << P.stretch
         << "  passes<=" << P.max_passes << "\n";
    cout << "============================================================================\n";

    // ---- PART 1: verification on the 8-node network ----
    RoadNetwork vnet = InstanceGenerator::createVerificationNetwork();
    auto vreqs = InstanceGenerator::createVerificationRequests();
    auto algos = standardAlgorithms(P);

    // Sanity check the verification instance itself
    bool allValid = true;
    for (size_t t = 0; t < vreqs.size(); ++t) {
        for (auto& a : algos) {
            StepResult r = a.run(vnet, vreqs[t], static_cast<int>(t + 1));
            if (!validateResult(vnet, r, capFor(a.name, P))) {
                cerr << "!!! VALIDATION FAILURE: verification instance t=" << (t + 1)
                     << " alg=" << a.name << "\n";
                allValid = false;
            }
        }
    }
    cout << " Verification-instance checks: " << (allValid ? "ALL PASSED" : "FAILED") << "\n";

    for (size_t t = 0; t < vreqs.size(); ++t) {
        if (detail) {
            for (auto& a : algos)
                printDetailedStepReport(vnet, a.run(vnet, vreqs[t], static_cast<int>(t + 1)), a.name);
        } else {
            printDetailedStepReport(
                vnet, TrafficRouter::solveMPCALB(vnet, vreqs[t], static_cast<int>(t + 1), P), "MP-CALB");
        }
    }
    cout << "\n============================================================================\n";
    cout << "  VERIFICATION INSTANCE: ALL ALGORITHMS, t = 1..3\n";
    cout << "============================================================================\n";
    cout << " Step | Alg      | Max load | Avg load | Load var | Total dist | ms\n";
    cout << "------+----------+----------+----------+----------+------------+------\n";
    for (size_t t = 0; t < vreqs.size(); ++t) {
        for (auto& a : algos) {
            auto m = a.run(vnet, vreqs[t], static_cast<int>(t + 1)).metrics;
            cout << "  t=" << (t + 1) << " | " << setw(8) << left << a.name
                 << " | " << setw(8) << right << fmt(m.max_load, 1)
                 << " | " << setw(8) << fmt(m.avg_load, 2)
                 << " | " << setw(8) << fmt(m.load_variance, 1)
                 << " | " << setw(10) << fmt(m.total_distance, 1)
                 << " | " << fmt(m.runtime_ms, 3) << "\n";
        }
        cout << "------+----------+----------+----------+----------+------------+------\n";
    }

    // ---- PART 1b (optional): per-step report on a GENERATED network ----
    if (generated) {
        ScaleConfig g("Small, grid", Family::GRID, 20, 0, 5, 8, 1001ULL);
        RoadNetwork gnet = makeNetwork(g, g.seed);
        auto greqs = InstanceGenerator::generateRequests(gnet.num_vertices, g.T, g.R, g.seed + 555ULL, HOT_FRAC);
        cout << "\n============================================================================\n";
        cout << "  GENERATED NETWORK: V=" << gnet.num_vertices << ", E=" << gnet.edges.size()
             << ", R=" << g.R << " requests/step, T=" << g.T << " time steps\n";
        cout << "============================================================================\n";
        for (int t = 0; t < g.T; ++t) {
            StepResult r = TrafficRouter::solveMPCALB(gnet, greqs[t], t + 1, P);
            printDetailedStepReport(gnet, r, "MP-CALB");
        }
    }

    // ---- PART 2: multi-scale benchmark ----
    vector<ScaleConfig> scales = {
        {"Small, grid",             Family::GRID,       20,    0, 5,   8, 1001ULL},
        {"Medium, grid",            Family::GRID,      100,    0, 5,  25, 2002ULL},
        {"Medium, geometric kNN",   Family::GEOMETRIC, 100,    0, 5,  25, 2004ULL},
        {"Medium, random chords",   Family::RANDOM,    100,  280, 5,  25, 2003ULL},
        {"Large, grid",             Family::GRID,      500,    0, 4, 100, 3003ULL},
    };
    if (!quick) scales.push_back({"X-Large, grid", Family::GRID, 1500, 0, 2, 250, 4004ULL});
    runBenchmark(scales, P, num_seeds, csv);

    // ---- PART 3: sweeps ----
    const ScaleConfig& med = scales[1];
    vector<pair<string, MPParams>> alpha_cfgs;
    for (double a : {0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0}) {
        MPParams q = P;
        q.alpha = a;
        alpha_cfgs.push_back({"a=" + fmt(a, 2), q});
    }
    runSweep("TRADE-OFF SWEEP over alpha (stretch cap fixed)", alpha_cfgs, med, num_seeds);

    vector<pair<string, MPParams>> stretch_cfgs;
    for (double s : {1.05, 1.1, 1.25, 1.5, 2.0, 1e9}) {
        MPParams q = P;
        q.alpha = 0.5;
        q.stretch = s;
        stretch_cfgs.push_back({s > 1e8 ? "no cap" : "x" + fmt(s, 2), q});
    }
    runSweep("STRETCH-CAP SWEEP (alpha = 0.5)", stretch_cfgs, med, num_seeds);

    vector<pair<string, MPParams>> abl_cfgs;
    {
        MPParams q = P;
        abl_cfgs.push_back({"full", q});
        q = P; q.refine = false;
        abl_cfgs.push_back({"no-ref", q});
        q = P; q.platoons = 1;
        abl_cfgs.push_back({"1 plat.", q});
        q = P; q.platoons = 3;
        abl_cfgs.push_back({"3 plat.", q});
        q = P; q.platoons = 12;
        abl_cfgs.push_back({"12 plat", q});
    }
    runSweep("ABLATION: refinement and platoon count (alpha = 0.25)", abl_cfgs, med, num_seeds);

    // ---- PART 4: scalability ----
    if (!quick) runScalability(P);

    // ---- PART 5: optimality check ----
    runOptimalityCheck(scales, P, num_seeds);

    if (csv) cout << "\nBenchmark results written to " << csv_path << "\n";
    return 0;
}
