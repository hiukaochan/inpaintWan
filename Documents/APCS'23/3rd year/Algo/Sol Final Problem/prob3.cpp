#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    cin >> m >> n;
    vector<long long> a(m), b(n);
    for (int i = 0; i < m; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> b[i];

    // sort b keeping original 1-based index
    vector<pair<long long, int>> bs(n);
    for (int i = 0; i < n; i++) bs[i] = {b[i], i + 1};
    sort(bs.begin(), bs.end());

    long long best = LLONG_MAX;
    int ans_i = -1, ans_j = -1;

    for (int i = 0; i < m; i++) {
        long long target = -a[i];
        int pos = (int)(lower_bound(bs.begin(), bs.end(), make_pair(target, INT_MIN)) - bs.begin());

        for (int p = pos - 1; p <= pos; p++) {
            if (p < 0 || p >= n) continue;
            long long val = abs(a[i] + bs[p].first);
            if (val < best) {
                best = val;
                ans_i = i + 1;
                ans_j = bs[p].second;
            }
        }
    }
    cout << ans_i << " " << ans_j << "\n";
    return 0;
}
