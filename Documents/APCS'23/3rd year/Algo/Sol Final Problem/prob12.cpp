#include <bits/stdc++.h>
using namespace std;

int main() {
    long long S;
    cin >> S;

    for (long long a = (long long)sqrt((double)S); a >= 0; a--) {
        long long rem = S - a * a;
        long long b = (long long)sqrt((double)rem);
        if (b * b == rem) {
            // vertices: (0,0), (a,b), (a-b, b+a), (-b, a)
            cout << 0 << " " << 0 << "\n";
            cout << a << " " << b << "\n";
            cout << a - b << " " << b + a << "\n";
            cout << -b << " " << a << "\n";
            return 0;
        }
    }
    cout << "Impossible\n";
    return 0;
}
