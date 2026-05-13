#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> adj[1005];
long long sz[1005];
long long ans = 0;

void dfs(int v, int par) {
    sz[v] = 1;
    for (int u : adj[v]) {
        if (u == par) continue;
        dfs(u, v);
        sz[v] += sz[u];
        ans += sz[u] * (n - sz[u]);
    }
}

int main() {
    cin >> n;
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs(1, 0);
    cout << ans << "\n";
    return 0;
}
