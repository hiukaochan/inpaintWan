#include <bits/stdc++.h>
using namespace std;

long long merge_count(vector<long long>& a, int l, int mid, int r) {
    vector<long long> tmp;
    int i = l, j = mid + 1;
    long long cnt = 0;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) {
            tmp.push_back(a[i++]);
        } else {
            cnt += (mid - i + 1);
            tmp.push_back(a[j++]);
        }
    }
    while (i <= mid) tmp.push_back(a[i++]);
    while (j <= r)   tmp.push_back(a[j++]);
    for (int k = l; k <= r; k++) a[k] = tmp[k - l];
    return cnt;
}

long long merge_sort(vector<long long>& a, int l, int r) {
    if (l >= r) return 0;
    int mid = (l + r) / 2;
    long long cnt = merge_sort(a, l, mid) + merge_sort(a, mid + 1, r);
    cnt += merge_count(a, l, mid, r);
    return cnt;
}

int main() {
    int n;
    cin >> n;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    cout << merge_sort(a, 0, n - 1) << "\n";
    return 0;
}
