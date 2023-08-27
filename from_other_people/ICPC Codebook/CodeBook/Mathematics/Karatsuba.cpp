void convo(int a[], int b[], int res[], int h1, int h2, int n){
	if (n <= 8){
		for (int i = h1; i < h1 + n; i++) {
			for (int j = h1; j < h1 + n; j++) {
				add(res[i + j], prod(a[i], b[j]));
			}
		}
	} else {
		const int mid = n >> 1;
		int atmp[mid], btmp[mid], E[n + 1];
		memset(E, 0, sizeof E);
		for(int i = h1; i < h1 + mid; ++i){
			atmp[i - h1] = sum(a[i], a[i + mid]);
			btmp[i - h1] = sum(b[i], b[i + mid]);
		}

		convo(atmp, btmp, E, 0, 0, mid);
		convo(a, b, res, h1, h2, mid);
		convo(a, b, res, h1 + mid, h2 + n, mid);

		for(int i = h2; i < h2 + mid; ++i){
			const int tmp = res[i + mid];
			add(res[i + mid], E[i - h2] - res[i] - res[i + 2 * mid]);
			add(res[i + 2 * mid], E[i - h2 + mid] - tmp - res[i + 3 * mid]);
		}
	}
}