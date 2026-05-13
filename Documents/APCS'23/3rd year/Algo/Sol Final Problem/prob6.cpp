#include <bits/stdc++.h>
using namespace std;

int main() {
    int k;
    cin >> k;
    vector<vector<int>> seqs(k);
    for (int i = 0; i < k; i++) {
        int sz;
        cin >> sz;
        seqs[i].resize(sz);
        for (int j = 0; j < sz; j++) cin >> seqs[i][j];
    }

    // min-heap: {value, seq_index, position}
    priority_queue<tuple<int,int,int>, vector<tuple<int,int,int>>, greater<>> pq;
    for (int i = 0; i < k; i++)
        if (!seqs[i].empty())
            pq.push({seqs[i][0], i, 0});

    bool first = true;
    while (!pq.empty()) {
        auto [val, i, pos] = pq.top(); pq.pop();
        if (!first) cout << " ";
        cout << val;
        first = false;
        if (pos + 1 < (int)seqs[i].size())
            pq.push({seqs[i][pos + 1], i, pos + 1});
    }
    cout << "\n";
    return 0;
}
