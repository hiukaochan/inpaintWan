#include <bits/stdc++.h>
using namespace std;

int main() {
    int k, n, m;
    cin >> k >> n >> m;
    vector<int> a(k+1), b(k+1);
    for (int i = 1; i <= k; i++) cin >> a[i] >> b[i];

    // dp[j][l] = max weight with j in Group1, l in Group2 (rolling over i)
    const int NEG_INF = -1e9;
    vector<vector<int>> dp(n+1, vector<int>(m+1, NEG_INF));
    dp[0][0] = 0;

    for (int i = 1; i <= k; i++) {
        // iterate backwards to avoid using cluster i twice
        for (int j = min(i, n); j >= 0; j--) {
            for (int l = min(i - j, m); l >= 0; l--) {
                if (dp[j][l] == NEG_INF) continue;
                // assign to group 1
                if (j + 1 <= n)
                    dp[j+1][l] = max(dp[j+1][l], dp[j][l] + a[i]);
                // assign to group 2
                if (l + 1 <= m)
                    dp[j][l+1] = max(dp[j][l+1], dp[j][l] + b[i]);
                // skip: dp[j][l] stays as is
            }
        }
    }

    cout << dp[n][m] << "\n";
    return 0;
}
