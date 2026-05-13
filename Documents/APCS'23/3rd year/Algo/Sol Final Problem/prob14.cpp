#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    // total area * 2 via shoelace
    long long total = 0;
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        total += x[i] * y[ni] - x[ni] * y[i];
    }
    total = abs(total); // total = 2 * area

    long long best_diff = LLONG_MAX;
    int ans_i = -1, ans_j = -1;

    for (int i = 0; i < n - 2; i++) {
        long long partial = 0;
        for (int j = i + 1; j < n - 1; j++) {
            // accumulate cross product for edge j -> j+1
            partial += x[j] * y[j+1] - x[j+1] * y[j];
            if (j >= i + 2) {
                // diagonal from i to j (0-indexed), i.e., vertices i and j
                // sub-polygon: i, i+1, ..., j  (closed back to i)
                long long area_A = abs(partial + x[j] * y[i] - x[i] * y[j]);
                // also add edge from i to i+1 start? No, partial already has i+1..j
                // We need the fan from i: add edge i -> i+1 first
                // Redo: partial here covers edges i+1->i+2, ..., j->j+1
                // We want sub-polygon i, i+1, ..., j => edges: i->i+1, i+1->i+2, ..., j-1->j, j->i
                // So we need to restart partial properly
                // Correction: let partial accumulate from edge (i -> i+1) at start of inner loop
                // Move partial init and accumulation before the if-check
                // This is already done below (restructured)
                (void)area_A; // placeholder
            }
        }
    }

    // Correct implementation: accumulate from edge i->i+1
    for (int i = 0; i < n - 2; i++) {
        long long partial = x[i] * y[i+1] - x[i+1] * y[i]; // edge i->i+1
        for (int j = i + 2; j <= n - 1; j++) {
            // add edge (j-1) -> j
            partial += x[j-1] * y[j] - x[j] * y[j-1];
            // diagonal (i, j): skip if it's just an edge (j=i+1 already skipped, j=n-1 with i=0 is also edge)
            if (i == 0 && j == n - 1) continue; // that's the edge (0, n-1)
            // close the sub-polygon: add edge j -> i
            long long area_A = abs(partial + x[j] * y[i] - x[i] * y[j]);
            long long diff = abs(area_A - (total - area_A));
            if (diff < best_diff) {
                best_diff = diff;
                ans_i = i + 1;
                ans_j = j + 1;
            }
        }
    }

    cout << ans_i << " " << ans_j << "\n";
    return 0;
}
