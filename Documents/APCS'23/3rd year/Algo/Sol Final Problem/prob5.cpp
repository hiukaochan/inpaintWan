#include <bits/stdc++.h>
using namespace std;

int main() {
    string A, B;
    cin >> A >> B;
    int la = A.size(), lb = B.size();

    // Build failure function
    vector<int> pi(lb, 0);
    for (int i = 1, k = 0; i < lb; i++) {
        while (k > 0 && B[k] != B[i]) k = pi[k - 1];
        if (B[k] == B[i]) k++;
        pi[i] = k;
    }

    // KMP scan
    vector<int> positions;
    for (int i = 0, k = 0; i < la; i++) {
        while (k > 0 && B[k] != A[i]) k = pi[k - 1];
        if (B[k] == A[i]) k++;
        if (k == lb) {
            positions.push_back(i - lb + 2); // 1-indexed
            k = pi[k - 1];
        }
    }

    for (int pos : positions) cout << pos << "\n";
    return 0;
}
