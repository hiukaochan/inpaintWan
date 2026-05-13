#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    cin >> m >> n;
    vector<vector<int>> grid(m, vector<int>(n));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) cin >> grid[i][j];

    int ans = 0;
    // run for v=0 and v=1
    for (int v = 0; v <= 1; v++) {
        vector<vector<int>> dp(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] != v) { dp[i][j] = 0; continue; }
                if (i == 0 || j == 0) dp[i][j] = 1;
                else dp[i][j] = min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
                ans = max(ans, dp[i][j]);
            }
        }
    }
    cout << ans << "\n";
    return 0;
}
