// verify_result.cpp
// 比较标准答案bin和程序输出bin的差异

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>

using namespace std;

void usage(const char* prog) {
    cout << "用法: " << prog << " <标准答案bin> <结果bin> [p] [q] [r]" << endl;
    cout << "  p, q, r 是张量维度，如果不提供则根据文件大小推断" << endl;
    cout << "示例: " << prog << " result_tc1.bin result_tc1.bin 4000 4000 4" << endl;
    exit(1);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage(argv[0]);
    }

    const char* ref_file = argv[1];
    const char* out_file = argv[2];

    int p = 0, q = 0, r = 0;

    // 如果提供了维度，使用提供的维度
    if (argc >= 6) {
        p = atoi(argv[3]);
        q = atoi(argv[4]);
        r = atoi(argv[5]);
    }

    // 读取标准答案
    ifstream ref_stream(ref_file, ios::binary);
    if (!ref_stream.is_open()) {
        cerr << "错误: 无法打开标准答案文件 " << ref_file << endl;
        return 1;
    }
    ref_stream.seekg(0, ios::end);
    size_t ref_size = ref_stream.tellg() / sizeof(double);
    ref_stream.seekg(0, ios::beg);
    vector<double> reference(ref_size);
    ref_stream.read(reinterpret_cast<char*>(reference.data()), ref_size * sizeof(double));
    ref_stream.close();

    // 读取结果
    ifstream out_stream(out_file, ios::binary);
    if (!out_stream.is_open()) {
        cerr << "错误: 无法打开结果文件 " << out_file << endl;
        return 1;
    }
    out_stream.seekg(0, ios::end);
    size_t out_size = out_stream.tellg() / sizeof(double);
    out_stream.seekg(0, ios::beg);
    vector<double> result(out_size);
    out_stream.read(reinterpret_cast<char*>(result.data()), out_size * sizeof(double));
    out_stream.close();

    // 检查尺寸匹配
    if (reference.size() != result.size()) {
        cerr << "错误: 文件大小不匹配" << endl;
        cerr << "  标准答案: " << reference.size() << " 元素 (" << reference.size() * sizeof(double) << " bytes)" << endl;
        cerr << "  结果: " << result.size() << " 元素 (" << result.size() * sizeof(double) << " bytes)" << endl;
        return 1;
    }

    // 如果未提供维度，默认1D
    if (p == 0 || q == 0 || r == 0) {
            cerr << "警告: 无法推断维度，假设为 1D" << endl;
            p = reference.size();
            q = 1;
            r = 1;
    }

    cout << "校验配置:" << endl;
    cout << "  标准答案: " << ref_file << endl;
    cout << "  结果: " << out_file << endl;
    cout << "  维度: p=" << p << ", q=" << q << ", r=" << r << endl;
    cout << "  元素数: " << reference.size() << endl;
    cout << endl;

    // 校验
    const double tol = 1e-8;
    double max_error = 0.0;
    double max_rel_error = 0.0;
    int error_count = 0;
    const int max_report = 20;

    for (size_t i = 0; i < reference.size(); ++i) {
        double diff = result[i] - reference[i];
        double err = fabs(diff);
        double rel_err = (reference[i] != 0.0) ? err / fabs(reference[i]) : err;

        if (err > max_error) {
            max_error = err;
        }
        if (rel_err > max_rel_error) {
            max_rel_error = rel_err;
        }

        if (err > tol || isnan(diff)) {
            error_count++;
            if (error_count <= max_report) {
                int pi = (i / (q * r)) % p;
                int qi = (i / r) % q;
                int ri = i % r;
                printf("  误差 at [%zu] (p=%d,q=%d,r=%d): result=%.6e, ref=%.6e, diff=%.6e, rel=%.6e\n",
                       i, pi, qi, ri, result[i], reference[i], err, rel_err);
            }
        }
    }

    cout << "========================================" << endl;
    if (error_count == 0) {
        cout << "结果: PASS" << endl;
    } else {
        cout << "结果: FAIL" << endl;
    }
    cout << "  误差元素数: " << error_count << " / " << reference.size() << endl;
    cout << "  最大绝对误差: " << scientific << max_error << defaultfloat << endl;
    cout << "  最大相对误差: " << scientific << max_rel_error << defaultfloat << endl;
    cout << "  容差: " << scientific << tol << defaultfloat << endl;

    return (error_count == 0) ? 0 : 1;
}
