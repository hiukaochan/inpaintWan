#include <bits/stdc++.h>
using namespace std;

int main() {
    double x1, y1, x2, y2, R;
    cin >> x1 >> y1 >> x2 >> y2 >> R;

    double d = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    const double PI = acos(-1.0);
    double uni;

    if (d >= 2 * R) {
        uni = 2 * PI * R * R;
    } else if (d == 0) {
        uni = PI * R * R;
    } else {
        double alpha = 2 * acos(d / (2 * R));
        double segment = R * R * (alpha - sin(alpha)) / 2.0;
        double I = 2 * segment;
        uni = 2 * PI * R * R - I;
    }

    cout << fixed << setprecision(3) << uni << "\n";
    return 0;
}
