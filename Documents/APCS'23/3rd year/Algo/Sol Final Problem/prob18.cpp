#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<int> start(n);
    for (int i = 0; i < n; i++) cin >> start[i];

    vector<int> sorted_perm(n);
    iota(sorted_perm.begin(), sorted_perm.end(), 1);

    if (start == sorted_perm) { cout << 0 << "\n"; return 0; }

    map<vector<int>, int> dist;
    queue<vector<int>> q;
    dist[start] = 0;
    q.push(start);

    while (!q.empty()) {
        vector<int> cur = q.front(); q.pop();
        int d = dist[cur];
        // try all prefix reversals of length 2..n
        for (int len = 2; len <= n; len++) {
            vector<int> nxt = cur;
            reverse(nxt.begin(), nxt.begin() + len);
            if (!dist.count(nxt)) {
                dist[nxt] = d + 1;
                if (nxt == sorted_perm) {
                    cout << d + 1 << "\n";
                    return 0;
                }
                q.push(nxt);
            }
        }
    }
    return 0;
}
