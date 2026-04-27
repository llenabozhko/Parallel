#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <tuple>
#include <vector>

using namespace std;

const int amount_rows = 5000; // Increased size to see real parallel benefits
const int amount_cols = 5000;

// Use heap for large arrays to avoid stack overflow
int** arr;

void init();
long long all_rows_total_sum(int amount_threads);
pair<long long, int> min_row_sum(int amount_threads);

int main() {
    omp_set_nested(1); // Keep this to allow sections + for loops
    init();

    for (int threads : {1, 2, 4, 8}) {
        cout << "--- Testing with " << threads << " threads ---\n";

        long long total = 0;
        pair<long long, int> min_res;

        double start = omp_get_wtime();

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                total = all_rows_total_sum(threads);
            }

            #pragma omp section
            {
                min_res = min_row_sum(threads);
            }
        }

        double end = omp_get_wtime();

        cout << "[✓] Total Sum: " << total << endl;
        cout << "[✓] Min Row Sum: " << min_res.first << " [Row: " << min_res.second << "]\n";
        cout << "[i] Execution Time: " << end - start << "s\n\n";
    }

    // Memory cleanup
    for(int i = 0; i < amount_rows; ++i) delete[] arr[i];
    delete[] arr;

    return 0;
}

void init() {
    arr = new int*[amount_rows];
    for(int i = 0; i < amount_rows; ++i) arr[i] = new int[amount_cols];

    srand(time(NULL));
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < amount_rows; ++i) {
        for (int j = 0; j < amount_cols; ++j) {
            arr[i][j] = rand() % 100; // Constrained for easier debugging
        }
    }
}

long long all_rows_total_sum(int amount_threads) {
    long long total_sum = 0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:total_sum) num_threads(amount_threads)
    for (int i = 0; i < amount_rows; ++i) {
        for (int j = 0; j < amount_cols; ++j) {
            total_sum += arr[i][j];
        }
    }

    double end = omp_get_wtime();
    cout << "[Section 1] Sum calculated in " << end - start << "s\n";
    return total_sum;
}

pair<long long, int> min_row_sum(int amount_threads) {
    long long min_val = -1; 
    int min_idx = 0;
    double start = omp_get_wtime();

    #pragma omp parallel num_threads(amount_threads)
    {
        long long local_min = -1;
        int local_idx = 0;

        #pragma omp for
        for (int i = 0; i < amount_rows; ++i) {
            long long current_row_sum = 0;
            for (int j = 0; j < amount_cols; ++j) {
                current_row_sum += arr[i][j];
            }

            if (local_min == -1 || current_row_sum < local_min) {
                local_min = current_row_sum;
                local_idx = i;
            }
        }

        // Only use critical section once per thread at the end
        #pragma omp critical
        {
            if (min_val == -1 || local_min < min_val) {
                min_val = local_min;
                min_idx = local_idx;
            }
        }
    }

    double end = omp_get_wtime();
    cout << "[Section 2] Min row found in " << end - start << "s\n";
    return {min_val, min_idx};
}