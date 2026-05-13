#include <bits/stdc++.h>
using namespace std;

void Sieve(vector<bool>& is_prime, int N) {
    is_prime[0] = is_prime[1] = false;
    for (int p = 2; p * p <= N; p++)
        if (is_prime[p])
            for (int m = p * p; m <= N; m += p)
                is_prime[m] = false;
}

int main() {
    int n;
    cin >> n;

    int N = 2 * n;
    vector<bool> is_prime(N + 1, true);
    Sieve(is_prime, N);

    long long f = 0;
    for (int i = 2; i <= n; i++) {
        for (int p = 2; p <= i; p++) {
            int q = 2 * i - p;
            if (is_prime[p] && is_prime[q])
                f++;
        }
    }
    cout << f << "\n";
    return 0;
}
