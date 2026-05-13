#include <bits/stdc++.h>
using namespace std;

// Compute factorials; use long long (safe up to n=20)
long long fact[21];

void precompute(int n) {
    fact[0] = 1;
    for (int i = 1; i <= n; i++) fact[i] = fact[i-1] * i;
}

// Given permutation p (1-indexed, 1..n), find its rank (1-indexed)
long long perm_to_rank(vector<int>& p, int n) {
    vector<int> available;
    for (int i = 1; i <= n; i++) available.push_back(i);
    long long rank = 1;
    for (int i = 0; i < n; i++) {
        int idx = (int)(find(available.begin(), available.end(), p[i]) - available.begin());
        rank += (long long)idx * fact[n - 1 - i];
        available.erase(available.begin() + idx);
    }
    return rank;
}

// Given rank y (1-indexed), find the permutation (1-indexed elements)
vector<int> rank_to_perm(long long y, int n) {
    vector<int> available;
    for (int i = 1; i <= n; i++) available.push_back(i);
    vector<int> perm;
    long long rem = y - 1;
    for (int i = n - 1; i >= 0; i--) {
        long long d = rem / fact[i];
        rem %= fact[i];
        perm.push_back(available[d]);
        available.erase(available.begin() + d);
    }
    return perm;
}

int main() {
    int n;
    cin >> n;
    precompute(n);

    // Task 1: read permutation, output rank
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    cout << perm_to_rank(p, n) << "\n";

    // Task 2: read rank, output permutation
    long long y;
    cin >> y;
    vector<int> q = rank_to_perm(y, n);
    for (int i = 0; i < n; i++) cout << q[i] << " \n"[i==n-1];

    return 0;
}
