#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

const int amount_rows = 1000;
const int amount_cols = 1000;
int arr[amount_rows][amount_cols];
int amount_threads = 8;

void init() {
    srand(time(NULL));
    cout << "- Init started" << endl;
    
    double start = omp_get_wtime();
    #pragma omp parallel for num_threads(amount_threads)
    for (int i = 0; i < amount_rows; ++i) {
        for (int j = 0; j < amount_cols; ++j) {
            arr[i][j] = rand() % 100;
        }
    }
    double end = omp_get_wtime();
    cout << "- Init finished in " << (end - start) << "s" << endl;
}

long long all_rows_total_sum(int threads) {
    long long sum = 0;
    double start = omp_get_wtime();
    
    #pragma omp parallel for reduction(+:sum) num_threads(threads)
    for (int i = 0; i < amount_rows; ++i) {
        for (int j = 0; j < amount_cols; ++j) {
            sum += arr[i][j];
        }
    }
    
    double end = omp_get_wtime();
    cout << "[SUM] Result: " << sum << " [" << (end - start) << "s] with " << threads << " threads" << endl;
    return sum;
}

int main() {
    cout << "Starting OpenMP test..." << endl;
    init();
    
    // Test with different thread counts
    for (int t : {1, 2, 4, 8}) {
        cout << "\n--- Testing with " << t << " threads ---" << endl;
        all_rows_total_sum(t);
    }
    
    cout << "\nDone!" << endl;
    return 0;
}
