// Decreasing Insertion, Query Min
struct CHT {
      vector<long long> a, b;

      bool cross(int i, int j, int k) {
            return 1.d*(a[j] - a[i])*(b[k] - b[i]) >= 1.d*(a[k] - a[i])*(b[j] - b[i]);
      }

      void add(long long A, long long B) {
            a.push_back(A);
            b.push_back(B);

            while (a.size() > 2 && cross(a.size() - 3, a.size() - 2, a.size() - 1)) {
            a.erase(a.end() - 2);
			b.erase(b.end() - 2);
		}
      }

      long long query(long long x) {
            int l = 0, r = a.size() - 1;

            while (l < r) {
                  int mid = l + (r - l)/2;
			long long f1 = a[mid] * x + b[mid];
			long long f2 = a[mid + 1] * x + b[mid + 1];

			if (f1 > f2) l = mid + 1;
			else r = mid;
            }

            return a[l]*x + b[l];
      }
};