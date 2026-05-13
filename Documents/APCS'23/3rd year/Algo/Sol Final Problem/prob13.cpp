#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    cin >> n;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];

    int half = (n - 2) / 2;

    for (int i = 0; i < n; i++) {
        // sort other points by polar angle around i
        vector<pair<double, int>> angles;
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double ang = atan2((double)(y[j] - y[i]), (double)(x[j] - x[i]));
            angles.push_back({ang, j});
        }
        sort(angles.begin(), angles.end());

        // cross product to count left-side points initially
        // count points strictly left of direction from i to angles[0].second
        int cnt = 0;
        {
            int q = angles[0].second;
            long long dx = x[q] - x[i], dy = y[q] - y[i];
            for (int j = 0; j < n; j++) {
                if (j == i || j == q) continue;
                long long cx = x[j] - x[i], cy = y[j] - y[i];
                // cross product: dx*cy - dy*cx > 0 means j is to the left
                if (dx * cy - dy * cx > 0) cnt++;
            }
        }

        for (int t = 0; t < n - 1; t++) {
            if (cnt == half) {
                cout << i + 1 << " " << angles[t].second + 1 << "\n";
                return 0;
            }
            // rotating past angles[t]: that point moves from front to back
            // points that were to the left of direction i->angles[t]
            // after passing angles[t], that point is now "behind" us
            // As we rotate CCW past angles[t], cnt changes:
            // The point angles[t].second was on the boundary (on the line),
            // now it goes to the right half-plane, so cnt decreases by
            // the number of points that cross over.
            // Simpler: after rotating to next angle, recount
            // Actually use the sweep properly:
            // when we advance from angles[t] to angles[t+1],
            // the point angles[t] moves from "ahead" to "behind" (right side)
            // This reduces cnt by the number of new "right" points.
            // Simple O(1) update: rotating past point angles[t]:
            //   it goes from the current half to the opposite, cnt-- (it was not counted anyway since it's ON the line)
            // Actually the count of points STRICTLY LEFT of line i->angles[t+1]:
            // = (points strictly left of i->angles[t])
            //   + (points that are between angle[t] and angle[t+1] exclusively = 0, since we sort)
            //   - (if angles[t] point crosses to right as we advance)
            // Proper update: as we sweep past angles[t], the point moves to
            // the right half-plane, so all points in (angles[t], angles[t]+pi)
            // that are now on left... this is complex; just do O(n) recount per step for simplicity
            if (t + 1 < n - 1) {
                int q = angles[t+1].second;
                long long dx = x[q] - x[i], dy = y[q] - y[i];
                cnt = 0;
                for (int j = 0; j < n; j++) {
                    if (j == i || j == q) continue;
                    long long cx = x[j] - x[i], cy = y[j] - y[i];
                    if (dx * cy - dy * cx > 0) cnt++;
                }
            }
        }
    }
    return 0;
}
