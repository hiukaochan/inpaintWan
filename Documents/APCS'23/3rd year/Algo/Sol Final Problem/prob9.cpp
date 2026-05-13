#include <bits/stdc++.h>
using namespace std;

int main() {
    string s, p;
    cin >> s >> p;
    int ls = s.size(), lp = p.size();

    // dp[i][j] = can s[0..i-1] be matched by p[0..j-1]
    vector<vector<bool>> dp(ls + 1, vector<bool>(lp + 1, false));
    dp[0][0] = true;

    // empty string vs pattern
    for (int j = 2; j <= lp; j++)
        if (p[j-1] == '*') dp[0][j] = dp[0][j-2];

    for (int i = 1; i <= ls; i++) {
        for (int j = 1; j <= lp; j++) {
            if (p[j-1] == '*') {
                // zero occurrences of p[j-2]
                dp[i][j] = (j >= 2 && dp[i][j-2]);
                // one or more: s[i-1] matches p[j-2]
                if (j >= 2 && (p[j-2] == '.' || p[j-2] == s[i-1]))
                    dp[i][j] = dp[i][j] || dp[i-1][j];
            } else {
                bool match = (p[j-1] == '.' || p[j-1] == s[i-1]);
                dp[i][j] = dp[i-1][j-1] && match;
            }
        }
    }

    cout << (dp[ls][lp] ? "true" : "false") << "\n";
    return 0;
}
