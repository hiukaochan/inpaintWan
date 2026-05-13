#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, k;
    cin >> n >> k;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    int best = 0;
    for (int i = 0; i <= n - k; i++) {
        int g = a[i];
        for (int j = i + 1; j < i + k; j++)
            g = gcd(g, a[j]);
        best = max(best, g);
    }
    cout << best << "\n";
    return 0;
}
