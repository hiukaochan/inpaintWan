#include <bits/stdc++.h>
using namespace std;

// Union-Find
struct DSU {
    vector<int> parent, rank_;
    DSU(int n) : parent(n), rank_(n, 0) { iota(parent.begin(), parent.end(), 0); }
    int find(int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) rank_[a]++;
        return true;
    }
    bool same(int a, int b) { return find(a) == find(b); }
};

int N, M, K;
struct Edge { int u, v; long long w; };
vector<Edge> edges;

// Find path from u to v in tree (given as adjacency list), return tree edge indices on path
vector<int> tree_path(int u, int v, vector<vector<pair<int,int>>>& tree_adj) {
    // BFS
    int n = tree_adj.size();
    vector<int> par(n, -1), par_edge(n, -1);
    vector<bool> vis(n, false);
    queue<int> q;
    q.push(u); vis[u] = true;
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        if (cur == v) break;
        for (auto [nb, eidx] : tree_adj[cur]) {
            if (!vis[nb]) { vis[nb] = true; par[nb] = cur; par_edge[nb] = eidx; q.push(nb); }
        }
    }
    vector<int> path;
    int cur = v;
    while (cur != u) { path.push_back(par_edge[cur]); cur = par[cur]; }
    return path;
}

int main() {
    cin >> N >> M >> K;
    edges.resize(M);
    for (int i = 0; i < M; i++) cin >> edges[i].u >> edges[i].v >> edges[i].w;

    // Sort edges by weight for Kruskal
    vector<int> order(M);
    iota(order.begin(), order.end(), 0);
    sort(order.begin(), order.end(), [](int a, int b){ return edges[a].w < edges[b].w; });

    // Kruskal MST
    DSU dsu(N + 1);
    vector<bool> in_mst(M, false);
    long long W0 = 0;
    for (int idx : order) {
        if (dsu.unite(edges[idx].u, edges[idx].v)) {
            in_mst[idx] = true;
            W0 += edges[idx].w;
        }
    }

    if (K == 1) { cout << W0 << "\n"; return 0; }

    // Build MST adjacency list (using sorted edge indices as edge IDs)
    vector<vector<pair<int,int>>> tree_adj(N + 1);
    for (int i = 0; i < M; i++) {
        if (in_mst[i]) {
            tree_adj[edges[i].u].push_back({edges[i].v, i});
            tree_adj[edges[i].v].push_back({edges[i].u, i});
        }
    }

    // PQ entry: (weight, non-tree edge added, tree edge removed, current tree as set)
    // Store tree as sorted vector of edge indices (sorted by their position in 'order')
    using State = tuple<long long, int, int, vector<bool>>;
    auto cmp = [](const State& a, const State& b){ return get<0>(a) > get<0>(b); };
    priority_queue<State, vector<State>, decltype(cmp)> pq(cmp);

    // Seed PQ with all 1-swap neighbors of MST
    // For each non-tree edge e, find cycle in MST, try swapping each tree edge on the cycle
    for (int i = 0; i < M; i++) {
        if (in_mst[i]) continue;
        vector<int> path = tree_path(edges[i].u, edges[i].v, tree_adj);
        for (int et : path) {
            long long w_new = W0 + edges[i].w - edges[et].w;
            vector<bool> new_tree = in_mst;
            new_tree[i] = true;
            new_tree[et] = false;
            pq.push({w_new, i, et, new_tree});
        }
    }

    int count = 1;
    while (!pq.empty()) {
        auto [w, e_added, e_removed, cur_tree] = pq.top(); pq.pop();
        count++;
        if (count == K) { cout << w << "\n"; return 0; }

        // Rebuild tree adjacency for cur_tree
        vector<vector<pair<int,int>>> cur_adj(N + 1);
        for (int i = 0; i < M; i++) {
            if (cur_tree[i]) {
                cur_adj[edges[i].u].push_back({edges[i].v, i});
                cur_adj[edges[i].v].push_back({edges[i].u, i});
            }
        }

        // Generate 1-swap neighbors: only non-tree edges with index > e_added (in sorted order)
        // to avoid duplicates
        int pos_e = (int)(find(order.begin(), order.end(), e_added) - order.begin());
        for (int pi = pos_e + 1; pi < M; pi++) {
            int i = order[pi];
            if (cur_tree[i]) continue;
            vector<int> path = tree_path(edges[i].u, edges[i].v, cur_adj);
            for (int et : path) {
                long long w_new = w + edges[i].w - edges[et].w;
                vector<bool> new_tree = cur_tree;
                new_tree[i] = true;
                new_tree[et] = false;
                pq.push({w_new, i, et, new_tree});
            }
        }
    }
    return 0;
}
