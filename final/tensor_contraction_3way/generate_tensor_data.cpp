// generate_tensor_data.cpp
// Generates input tensor data for the contraction program

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <mpi.h>

using namespace std;

vector<double> generate_tensor(size_t n, int seed = 12345)
{
    vector<double> data(n);
    mt19937_64 rng(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

void write_tensor_to_file(const string& filename, const vector<double>& data) {
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing" << endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
    file.close();
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    
    int rank, size, scale_factor;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 1) {
        if (rank == 0) {
            cerr << "Error: This generator requires exactly 1 MPI processes\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    string data_dir = "tensor_data";
    if (argc > 1) {
        scale_factor = atoi(argv[1]);
    }
    
    if (rank == 0) {
        cout << "Generating tensor data for test cases..." << endl;
        // 创建目录（简单处理，假设目录已存在）
    }
    
    // ============ Test Case 1 ============
    int p = 4000, q = 4000, t = 4000, u = 4000, s = 4, r = 4;
    p = p / scale_factor;
    q = q / scale_factor;
    t = t / scale_factor;
    u = u / scale_factor;

    const int p_local = p / size;
    const int q_local = q / size;
    
    // A_local: [s, t, p_local]
    vector<double> A_local_tc1 = generate_tensor(s * t * p_local, 12345 + rank);
    string a_filename = data_dir + "/A_tc1_" + to_string(scale_factor) + ".bin";
    write_tensor_to_file(a_filename, A_local_tc1);
    
    // B_local: [s, u, q_local]
    vector<double> B_local_tc1 = generate_tensor(s * u * q_local, 34567 + rank);
    string b_filename = data_dir + "/B_tc1_" + to_string(scale_factor) + ".bin";
    write_tensor_to_file(b_filename, B_local_tc1);
    
    // C_full: [t, u, r] - only rank 0 generates
    if (rank == 0) {
        vector<double> C_full_tc1 = generate_tensor(t * u * r, 23456);
        string c_filename = data_dir + "/C_tc1_" + to_string(scale_factor) + ".bin";
        write_tensor_to_file(c_filename, C_full_tc1);
        cout << "Generated TC1 data for " << size << " ranks" << endl;
    }
    
    // ============ Test Case 2 ============
    int p2 = 8000, q2 = 4000, u2 = 4000, s2 = 8000, t2 = 2, r2 = 2;
    p2 = p2 / scale_factor;
    q2 = q2 / scale_factor;
    u2 = u2 / scale_factor;
    s2 = s2 / scale_factor;

    const int p_local2 = p2 / size;
    const int q_local2 = q2 / size;
    
    // A_local: [s, p_local, t]
    vector<double> A_local_tc2 = generate_tensor(s2 * p_local2 * t2, 12345 + rank + 100000);
    string a_filename2 = data_dir + "/A_tc2_" + to_string(scale_factor) + ".bin";
    write_tensor_to_file(a_filename2, A_local_tc2);
    
    // B_local: [t, q_local, u]
    vector<double> B_local_tc2 = generate_tensor(t2 * q_local2 * u2, 34567 + rank + 100000);
    string b_filename2 = data_dir + "/B_tc2_" + to_string(scale_factor) + ".bin";
    write_tensor_to_file(b_filename2, B_local_tc2);
    
    // C_full: [u, r, s] - only rank 0 generates
    if (rank == 0) {
        vector<double> C_full_tc2 = generate_tensor(u2 * r2 * s2, 23456 + 100000);
        string c_filename2 = data_dir + "/C_tc2_"  + to_string(scale_factor) + ".bin";
        write_tensor_to_file(c_filename2, C_full_tc2);
        cout << "Generated TC2 data for " << size << " ranks" << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "All tensor data generated successfully in " << data_dir << "/" << endl;
    }
    
    MPI_Finalize();
    return 0;
}