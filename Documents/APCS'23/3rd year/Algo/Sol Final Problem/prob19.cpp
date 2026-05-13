#include <bits/stdc++.h>
using namespace std;

int n;
vector<long long> a;

// cost to achieve minimum height h, with the deepest point at position j
// = sum_i max(0, a[i] - h - |i-j|)
long long cost_at_j(int j, long long h) {
    long long c = 0;
    for (int i = 0; i < n; i++) {
        long long diff = a[i] - h - abs(i - j);
        if (diff > 0) c += diff;
    }
    return c;
}

// COST(h) = min over all j of cost_at_j(j, h)
// Use ternary search since cost_at_j is convex in j
long long COST(long long h) {
    int lo = 0, hi = n - 1;
    while (hi - lo > 2) {
        int m1 = lo + (hi - lo) / 3;
        int m2 = hi - (hi - lo) / 3;
        if (cost_at_j(m1, h) < cost_at_j(m2, h))
            hi = m2;
        else
            lo = m1;
    }
    long long best = LLONG_MAX;
    for (int j = lo; j <= hi; j++)
        best = min(best, cost_at_j(j, h));
    return best;
}

int main() {
    cin >> n;
    long long T;
    cin >> T;
    a.resize(n);
    long long max_a = 0;
    for (int i = 0; i < n; i++) { cin >> a[i]; max_a = max(max_a, a[i]); }

    // binary search on h: find minimum h such that COST(h) <= T
    long long lo = -T, hi = max_a;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (COST(mid) <= T)
            hi = mid;
        else
            lo = mid + 1;
    }
    cout << lo << "\n";
    return 0;
}
