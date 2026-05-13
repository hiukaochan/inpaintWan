#include <bits/stdc++.h>
using namespace std;

int N, M;
vector<pair<int,int>> adj[100005]; // {neighbor, edge_index}
int disc[100005], low_[100005], sz[100005];
long long bridge_pairs[100005]; // indexed by edge_index; 0 if not a bridge
int timer_ = 0;

void dfs(int v, int par_edge) {
    disc[v] = low_[v] = timer_++;
    sz[v] = 1;
    for (auto [u, eidx] : adj[v]) {
        if (disc[u] == -1) {
            dfs(u, eidx);
            sz[v] += sz[u];
            low_[v] = min(low_[v], low_[u]);
            if (low_[u] > disc[v]) {
                // (v, u) is a bridge
                bridge_pairs[eidx] = (long long)sz[u] * (N - sz[u]);
            }
        } else if (eidx != par_edge) {
            low_[v] = min(low_[v], disc[u]);
        }
    }
}

int main() {
    cin >> N >> M;
    for (int i = 0; i < M; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        bridge_pairs[i] = 0;
    }

    memset(disc, -1, sizeof(disc));
    for (int i = 1; i <= N; i++)
        if (disc[i] == -1) dfs(i, -1);

    for (int i = 0; i < M; i++)
        cout << bridge_pairs[i] << "\n";
    return 0;
}
