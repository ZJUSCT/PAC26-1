// tensor_contraction_3way_opt.cpp
// Optimized 3-tensor contraction with proper memory layout for cache efficiency
// Key optimization: Reorder dimensions before contraction to ensure inner loops are contiguous

#include <mpi.h>
#include "tc_kernels.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <fstream>

using namespace std;

#ifdef DEBUG
#define DEBUG_PRINT(x) x
#else
#define DEBUG_PRINT(x)
#endif

// 写入张量数据到文件
void write_tensor_to_file(const string& filename, const vector<double>& data)
{
    ofstream file(filename, ios::binary);
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << " for writing" << endl;
        return;
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
    file.close();
}

/*****************************************************************************/
/************************* 3-Tensor Contraction with Optimized Layout ********/
/* Testcase1: A[s,t,p] x B[s,u,q] x C[t,u,r] -> (p,q,r)                       */
// p = 4000, q = 4000, t = 4000, u = 4000, s = 4, r = 4;
/*****************************************************************************/
void tensor_contract_3way_tc1_opt(vector<double>& A_local,  // input: A [s * t * p_local]
                                  vector<double>& B_local,  // input: B [s * u * q_local]
                                  vector<double>& C_full,  // input: C [t * u * r]
                                  vector<double>& D_buff,  // workspace
                                  vector<double>& C_local_out,  // output: [p_local * q * r]
                                  vector<double>& D_full,  // workspace: full D after Alltoallv
                                  MPI_Comm comm, int s, int t, int u, int p, int q, int r, int p_local, int q_local,
                                  int rank, int size)
{
    tcopt::tc1(A_local,B_local,C_full,C_local_out,comm,s,t,u,p,q,r,p_local,q_local,rank,size);
}

/*****************************************************************************/
/************************* 3-Tensor Contraction with Optimized Layout ********/
/* Testcase2: A[s,p,t] x B[t,q,u] x C[u,r,s] -> (p,q,r)                       */
// p = 8000, q = 4000, u = 4000, s = 8000, t = 2, r = 2;
/*****************************************************************************/
void tensor_contract_3way_tc2_opt(vector<double>& A_local,  // input: A [s * p_local * t]
                                  vector<double>& B_local,  // input: B [t * q_local * u]
                                  vector<double>& C_full,  // input: C [u * r * s]
                                  vector<double>& E_buff,  // workspace
                                  vector<double>& C_local_out,  // output: [p_local * q * r]
                                  vector<double>& E_full,  // workspace: full E after Alltoallv
                                  MPI_Comm comm, int s, int p, int t, int q, int u, int r, int p_local, int q_local,
                                  int rank, int size)
{
    tcopt::tc2(A_local,B_local,C_full,C_local_out,comm,s,p,t,q,u,r,p_local,q_local,rank,size);
}

// ========================
// Test Case 1: (s,t,p) x (s,u,q) x (t,u,r) -> (p,q,r)
// A[s,t,p], B[s,u,q], C[t,u,r] -> output[p,q,r]
// p=q=t=u=4000, s=r=4
// Distribution: A along p, B along q, C replicated
// Algorithm: B*C over u, then D*A over s,t with Alltoallv exchange
// ========================
void testcase1(MPI_Comm comm, const string& data_dir)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int p = 4000, q = 4000, t = 4000, u = 4000, s = 4, r = 4;

    const int p_local = p / size;
    const int q_local = q / size;

    if (rank == 0) {
        cout << "\nTest Case 1: (s,t,p) x (s,u,q) x (t,u,r) -> (p,q,r)"
             << ", shapes: (" << s << "," << t << "," << p << ") x (" << s << "," << u << "," << q << ") x (" << t
             << "," << u << "," << r << ") -> (" << p << "," << q << "," << r << ")\n";
        cout << "Optimization: Reorder layouts for contiguous inner loops\n";
    }

    vector<double> A_local(s * t * p_local);
    vector<double> B_local(s * u * q_local);
    vector<double> C_full(t * u * r);
    vector<double> D_buff, D_full;
    vector<double> C_local_out(p_local * q * r);

    // Read tensor data using MPI Subarray
    MPI_File fh;
    MPI_Datatype filetype_A, filetype_B, filetype_C;
    MPI_Status status;

    // A[s,t,p] - read p_local slice
    int A_sizes[3] = {s, t, p};
    int A_subsizes[3] = {s, t, p_local};
    int A_starts[3] = {0, 0, rank * p_local};
    MPI_Type_create_subarray(3, A_sizes, A_subsizes, A_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_A);
    MPI_Type_commit(&filetype_A);

    // B[s,u,q] - read q_local slice
    int B_sizes[3] = {s, u, q};
    int B_subsizes[3] = {s, u, q_local};
    int B_starts[3] = {0, 0, rank * q_local};
    MPI_Type_create_subarray(3, B_sizes, B_subsizes, B_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_B);
    MPI_Type_commit(&filetype_B);

    // C(p,q,r) - read ALL data
    int C_sizes[3] = {t, u, r};
    int C_subsizes[3] = {t, u, r};
    int C_starts[3] = {0, 0, 0};
    MPI_Type_create_subarray(3, C_sizes, C_subsizes, C_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_C);
    MPI_Type_commit(&filetype_C);

    // Read A
    string a_file = data_dir + "/A_tc1.bin";
    MPI_File_open(comm, a_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_A, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, A_local.data(), s * t * p_local, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    DEBUG_PRINT(cout << "rank " << rank << " tc1: A_local read from " << a_file << endl);

    // Read B
    string b_file = data_dir + "/B_tc1.bin";
    MPI_File_open(comm, b_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_B, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, B_local.data(), s * u * q_local, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    DEBUG_PRINT(cout << "rank " << rank << " tc1: B_local read from " << b_file << endl);

    // Read C full
    int count = 0;
    if (rank == 0) {
        count = t * u * r;
    }
    string c_file = data_dir + "/C_tc1.bin";
    MPI_File_open(comm, c_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_C, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, C_full.data(), count, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    DEBUG_PRINT(cout << "rank " << rank << " tc1: C_full read from " << c_file << endl);
    MPI_Type_free(&filetype_C);

    MPI_Type_free(&filetype_A);
    MPI_Type_free(&filetype_B);

    /*****************************************************************************/
    MPI_Barrier(comm);
    double start_time = MPI_Wtime();

    tensor_contract_3way_tc1_opt(A_local, B_local, C_full, D_buff, C_local_out, D_full, comm, s, t, u, p, q, r, p_local,
                                 q_local, rank, size);

    MPI_Barrier(comm);
    double end_time = MPI_Wtime();
    /*****************************************************************************/

    if (rank == 0) {
        // long long total_flops = 2LL * s * t * u * p * q * r;
        // double gflops = total_flops / ((end_time - start_time) * 1e9);
        cout << "rank " << rank << " time: " << (end_time - start_time) << " s" << endl;
    }

    // Gather and output
    vector<double> C_out_full;
    if (rank == 0) {
        C_out_full.resize(p * q * r);
    }

    vector<double> recv_buf;
    if (rank == 0) {
        recv_buf.resize(p_local * q * r * size);
    }

    MPI_Gather(C_local_out.data(), p_local * q * r, MPI_DOUBLE, recv_buf.data(), p_local * q * r, MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        string out_file = data_dir + "/result_tc1.bin";
        write_tensor_to_file(out_file, recv_buf);
        cout << "Result written to " << out_file << endl;
    }
}

// ========================
// Test Case 2: (s,p,t) x (t,q,u) x (u,r,s) -> (p,q,r)
// A[s,p,t], B[t,q,u], C[u,r,s] -> output[p,q,r]
// p=s=8000, q=u=4000, r=t=2
// Distribution: A along p, B along q, C replicated
// Algorithm: B*C over u, then E*A over s,t with Alltoallv exchange
// ========================
void testcase2(MPI_Comm comm, const string& data_dir)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int p = 8000, q = 4000, u = 4000, s = 8000, t = 2, r = 2;

    const int p_local = p / size;
    const int q_local = q / size;

    if (rank == 0) {
        cout << "\nTest Case 2: (s,p,t) x (t,q,u) x (u,r,s) -> (p,q,r)"
             << ", shapes: (" << s << "," << p << "," << t << ") x (" << t << "," << q << "," << u << ") x (" << u
             << "," << r << "," << s << ") -> (" << p << "," << q << "," << r << ")\n";
        cout << "Optimization: Reorder layouts for contiguous inner loops\n";
    }

    vector<double> A_local(s * p_local * t);
    vector<double> B_local(t * q_local * u);
    vector<double> C_full(u * r * s);
    vector<double> E_buff, E_full;
    vector<double> C_local_out(p_local * q * r);

    // Read tensor data using MPI Subarray
    MPI_File fh;
    MPI_Datatype filetype_A, filetype_B, filetype_C;
    MPI_Status status;

    // A[s,p,t] - read p_local slice
    int A_sizes[3] = {s, p, t};
    int A_subsizes[3] = {s, p_local, t};
    int A_starts[3] = {0, rank * p_local, 0};
    MPI_Type_create_subarray(3, A_sizes, A_subsizes, A_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_A);
    MPI_Type_commit(&filetype_A);

    // B[t,q,u] - read q_local slice
    int B_sizes[3] = {t, q, u};
    int B_subsizes[3] = {t, q_local, u};
    int B_starts[3] = {0, rank * q_local, 0};
    MPI_Type_create_subarray(3, B_sizes, B_subsizes, B_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_B);
    MPI_Type_commit(&filetype_B);

    // C(p,q,r) - read ALL data
    int C_sizes[3] = {u, r, s};
    int C_subsizes[3] = {u, r, s};
    int C_starts[3] = {0, 0, 0};
    MPI_Type_create_subarray(3, C_sizes, C_subsizes, C_starts, MPI_ORDER_C, MPI_DOUBLE, &filetype_C);
    MPI_Type_commit(&filetype_C);

    // Read A
    string a_file = data_dir + "/A_tc2.bin";
    MPI_File_open(comm, a_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_A, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, A_local.data(), s * p_local * t, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    // Read B
    string b_file = data_dir + "/B_tc2.bin";
    MPI_File_open(comm, b_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_B, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, B_local.data(), t * q_local * u, MPI_DOUBLE, &status);
    MPI_File_close(&fh);

    // Read C full
    int count = 0;
    if (rank == 0) {
        count = u * s * r;
    }
    string c_file = data_dir + "/C_tc2.bin";
    MPI_File_open(comm, c_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_DOUBLE, filetype_C, "native", MPI_INFO_NULL);
    MPI_File_read_at(fh, 0, C_full.data(), count, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    DEBUG_PRINT(cout << "rank " << rank << " tc2: C_full read from " << c_file << endl);

    MPI_Type_free(&filetype_A);
    MPI_Type_free(&filetype_B);
    MPI_Type_free(&filetype_C);

    /*****************************************************************************/
    /********************************* 计时区间开始 *******************************/
    /*****************************************************************************/
    MPI_Barrier(comm);
    double start_time = MPI_Wtime();

    tensor_contract_3way_tc2_opt(A_local, B_local, C_full, E_buff, C_local_out, E_full, comm, s, p, t, q, u, r, p_local,
                                 q_local, rank, size);

    MPI_Barrier(comm);
    double end_time = MPI_Wtime();
    /*****************************************************************************/
    /********************************* 计时区间结束 *******************************/
    /*****************************************************************************/

    if (rank == 0) {
        // long long total_flops = 2LL * s * p * t * q * u * r;
        // double gflops = total_flops / ((end_time - start_time) * 1e9);
        cout << "rank " << rank << " time: " << (end_time - start_time) << " s" << endl;
    }

    // Gather and output
    vector<double> C_out_full;
    if (rank == 0) {
        C_out_full.resize(p * q * r);
    }

    vector<double> recv_buf;
    if (rank == 0) {
        recv_buf.resize(p_local * q * r * size);
    }

    MPI_Gather(C_local_out.data(), p_local * q * r, MPI_DOUBLE, recv_buf.data(), p_local * q * r, MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        for (int rk = 0; rk < size; ++rk) {
            for (int pi = 0; pi < p_local; ++pi) {
                for (int qi = 0; qi < q; ++qi) {
                    for (int ri = 0; ri < r; ++ri) {
                        int src_idx = ((rk * p_local + pi) * q + qi) * r + ri;
                        int dst_p = rk * p_local + pi;
                        C_out_full[(dst_p * q + qi) * r + ri] = recv_buf[src_idx];
                    }
                }
            }
        }
        string out_file = data_dir + "/result_tc2.bin";
        write_tensor_to_file(out_file, C_out_full);
        cout << "Result written to " << out_file << endl;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string data_dir = "tensor_data";
    if (argc > 1) {
        data_dir = argv[1];
    }

    testcase1(MPI_COMM_WORLD, data_dir);
    testcase2(MPI_COMM_WORLD, data_dir);

    MPI_Finalize();
    return 0;
}
