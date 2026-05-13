# Algorithm Problems — Solutions Reference

22 problems, each in its own `.cpp` file. Compile with:
```
g++ -std=c++17 -O2 -I. -o probN probN.cpp
```

---

## Problem 1 — Max GCD of k Consecutive Elements
**File:** `prob1.cpp`  
**Algorithm:** Slide a window of exactly k elements across the array; compute the GCD of the window with the Euclidean algorithm; track the maximum over all windows.  
**Complexity:** O(n · k · log V)

---

## Problem 2 — Cumulative Goldbach Pair Count
**File:** `prob2.cpp`  
**Algorithm:** Sieve of Eratosthenes up to 2n to mark all primes; for each i from 2 to n, count unordered prime pairs (p, q) with p + q = 2i and p ≤ q; accumulate the running sum f[i] = f[i−1] + g[i].  
**Complexity:** O(n²)

---

## Problem 3 — Minimise |a_i + b_j|
**File:** `prob3.cpp`  
**Algorithm:** Sort array B (keeping original indices); for each a_i binary-search for the target −a_i in sorted B and check both the floor and ceiling candidates; keep the globally best pair.  
**Complexity:** O((m + n) log n)

---

## Problem 4 — Count Inversions
**File:** `prob4.cpp`  
**Algorithm:** Merge sort — during the merge step, when a left element exceeds a right element, all remaining left elements also form inversions with that right element; add them in O(1) instead of O(n).  
**Complexity:** O(n log n)

---

## Problem 5 — KMP Pattern Matching
**File:** `prob5.cpp`  
**Algorithm:** Build the failure function π for pattern B (longest proper prefix that is also a suffix); scan text A once with a single pointer k, falling back via π on mismatch — no character of A is ever revisited.  
**Complexity:** O(|A| + |B|)

---

## Problem 6 — Merge k Sorted Sequences
**File:** `prob6.cpp`  
**Algorithm:** Min-heap of size ≤ k holding one entry per non-exhausted sequence; repeatedly extract the global minimum, output it, and insert the next element from that sequence.  
**Complexity:** O(N log k)  where N = total elements

---

## Problem 7 — Trapping Rain Water
**File:** `prob7.cpp`  
**Algorithm:** Two linear passes: build L[i] = prefix maximum and R[i] = suffix maximum; water trapped at position i is max(0, min(L[i], R[i]) − h[i]); sum over all positions.  
**Complexity:** O(n)

---

## Problem 8 — N-Queens
**File:** `prob8.cpp`  
**Algorithm:** Backtracking column-by-column; three boolean arrays (row_used, diag1, diag2) provide O(1) conflict checks; on conflict skip the row, otherwise place, recurse, then undo.  
**Complexity:** O(n!)  with heavy pruning (n ≤ 9)

---

## Problem 9 — Regex Matching ('.' and '*')
**File:** `prob9.cpp`  
**Algorithm:** 2-D DP where dp[i][j] = true if s[1..i] is fully matched by p[1..j]; '*' is handled as either zero occurrences (skip two pattern chars) or one-or-more (extend a prior match).  
**Complexity:** O(|s| · |p|)

---

## Problem 10 — Largest All-Same Square
**File:** `prob10.cpp`  
**Algorithm:** 2-D grid DP run independently for v = 0 and v = 1; dp[i][j] = min(top, left, top-left) + 1 when grid[i][j] equals v, else 0; answer is the overall maximum.  
**Complexity:** O(m · n)

---

## Problem 11 — Optimal Cluster Assignment (Two Groups)
**File:** `prob11.cpp`  
**Algorithm:** 3-D DP rolled into 2-D: dp[j][l] = maximum weight using exactly j clusters in Group 1 and l in Group 2, iterated over clusters i from 1 to k with a reverse loop to avoid reuse.  
**Complexity:** O(k · n · m)

---

## Problem 12 — Construct Integer-Area Square
**File:** `prob12.cpp`  
**Algorithm:** Try integer a from ⌊√S⌋ down to 0; check whether rem = S − a² is a perfect square b²; if yes, output the four vertices (0,0), (a,b), (a−b, a+b), (−b, a); if no solution exists, output "Impossible".  
**Complexity:** O(√S)

---

## Problem 13 — Line Splitting n Points in Half
**File:** `prob13.cpp`  
**Algorithm:** For each pivot point i, sort the remaining n−1 points by polar angle around i; sweep the directed half-plane and recount the left-side points at each angle step; stop when the count equals (n−2)/2.  
**Complexity:** O(n² log n)

---

## Problem 14 — Convex Polygon Diagonal (Min Area Difference)
**File:** `prob14.cpp`  
**Algorithm:** Compute total doubled-area via the shoelace formula (integers only); for each starting vertex i, accumulate cross-products incrementally as the far endpoint j sweeps forward; track the diagonal with minimum |2·areaA − total|.  
**Complexity:** O(n²)

---

## Problem 15 — Union Area of Two Equal Circles
**File:** `prob15.cpp`  
**Algorithm:** Inclusion-exclusion: union = 2πR² − intersection; intersection = 2 × circular-segment area where the half-angle α = 2·arccos(d / 2R); handle d ≥ 2R (disjoint) and d = 0 (identical) as special cases.  
**Complexity:** O(1)

---

## Problem 16 — Tree Width (Sum of All Pairwise Distances)
**File:** `prob16.cpp`  
**Algorithm:** Root the tree; one DFS computes subtree sizes sz[]; each edge (parent, child) is used by exactly sz[child] × (n − sz[child]) vertex pairs; sum these contributions over all n−1 edges.  
**Complexity:** O(n)

---

## Problem 17 — Smallest 0/1-digit Multiple of n
**File:** `prob17.cpp`  
**Algorithm:** BFS on a remainder graph with n states (0 … n−1); start from state 1 (the digit "1"); each transition appends digit 0 or 1 giving new remainder (r×10+d) mod n; the first time state 0 is reached the number string is the answer.  
**Complexity:** O(n)

---

## Problem 18 — Minimum Prefix-Reversals to Sort
**File:** `prob18.cpp`  
**Algorithm:** BFS over the space of all permutations; each state is a permutation and each transition is a prefix reversal of length 2 … n; BFS guarantees the first time the sorted permutation is reached it is via the minimum number of steps.  
**Complexity:** O(n · n!)

---

## Problem 19 — Minimum Achievable Height with Budget T
**File:** `prob19.cpp`  
**Algorithm:** Binary search on the target height h; for each candidate h, ternary-search over the position j of the deepest point and evaluate cost(j, h) = Σ max(0, a[i] − h − |i−j|) in O(n); the minimum h where cost ≤ T is the answer.  
**Complexity:** O(n log V)  where V = max height + T

---

## Problem 20 — Permutation Rank (Encode & Decode)
**File:** `prob20.cpp`  
**Algorithm:** Factorial number system: encode — for each position i scan remaining elements to find d_i (count of smaller remaining elements), then rank += d_i × (n−i)!; decode — divide remainder by (n−i)! to find the digit d_i and pick the (d_i+1)-th available element.  
**Complexity:** O(n²)

---

## Problem 21 — K-th Minimum Spanning Tree
**File:** `prob21.cpp`  
**Algorithm:** Kruskal's algorithm for the 1st MST; seed a min-heap with all 1-swap neighbours (add one non-tree edge, remove one tree edge on the created cycle); canonical forward-only ordering of swaps ensures each spanning tree is generated exactly once; the K-th extraction is the answer.  
**Complexity:** O(M log M + K · M)

---

## Problem 22 — Pairs Disconnected by Each Edge Removal
**File:** `prob22.cpp`  
**Algorithm:** Tarjan's bridge-finding DFS: track discovery time disc[v] and low-link value low[v]; edge (u → v) is a bridge iff low[v] > disc[u]; for each bridge the number of disconnected pairs = sz[v] × (N − sz[v]); non-bridges output 0.  
**Complexity:** O(N + M)
