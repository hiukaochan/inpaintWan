#include <bits/stdc++.h>
using namespace std;

int n;
vector<int> placement;
vector<vector<int>> solutions;
bool row_used[10], diag1[20], diag2[20];

void solve(int col) {
    if (col == n) {
        solutions.push_back(placement);
        return;
    }
    for (int r = 0; r < n; r++) {
        if (row_used[r] || diag1[r - col + n] || diag2[r + col])
            continue;
        placement[col] = r;
        row_used[r] = diag1[r - col + n] = diag2[r + col] = true;
        solve(col + 1);
        row_used[r] = diag1[r - col + n] = diag2[r + col] = false;
    }
}

int main() {
    cin >> n;
    placement.resize(n);
    solve(0);

    cout << solutions.size() << "\n";
    for (auto& sol : solutions) {
        for (int col = 0; col < n; col++) {
            for (int r = 0; r < n; r++)
                cout << (sol[col] == r ? 'Q' : '.');
            cout << "\n";
        }
        cout << "\n";
    }
    return 0;
}
