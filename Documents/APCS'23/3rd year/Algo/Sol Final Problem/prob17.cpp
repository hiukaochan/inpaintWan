#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;

    vector<bool> visited(n, false);
    queue<pair<int, string>> q;

    int start = 1 % n;
    visited[start] = true;
    q.push({start, "1"});

    while (!q.empty()) {
        auto [r, num] = q.front(); q.pop();
        if (r == 0) {
            cout << num << "\n";
            return 0;
        }
        for (int d : {0, 1}) {
            int nr = (r * 10 + d) % n;
            if (!visited[nr]) {
                visited[nr] = true;
                q.push({nr, num + (char)('0' + d)});
            }
        }
    }
    return 0;
}
