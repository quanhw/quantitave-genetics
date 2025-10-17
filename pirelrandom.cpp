#define EIGEN_DONT_PARALLELIZE

#if defined(__GNUC__) || defined(__clang__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <vector>
#include <unordered_map>
#include <random>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <cmath>
#include <memory>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <utility>
#include <cstring>
#include <cstdlib>

#include <stdlib.h>

#ifdef _OPENMP
# include <omp.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
# pragma GCC diagnostic pop
#endif

using namespace Eigen;
using namespace std;

#define FNAME_LEN 256

// =============================================================================
// Memory-aligned allocator
// =============================================================================
template<typename T>
class AlignedAllocator {
public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    
    template<typename U>
    struct rebind {
        typedef AlignedAllocator<U> other;
    };
    
    AlignedAllocator() noexcept = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();
    
        void* p = nullptr;
        if (posix_memalign(&p, 64, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) noexcept {
        free(p);
    }
};

template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T>>;

// =============================================================================
// Performance Monitor
// =============================================================================
class PerformanceMonitor {
private:
    std::map<std::string, double> timings;
    std::map<std::string, int> counts;
    
public:
    void start(const std::string& name) {
        #ifdef _OPENMP
        timings[name + "_start"] = omp_get_wtime();
        #else
        timings[name + "_start"] = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        #endif
    }
    
    void stop(const std::string& name) {
        #ifdef _OPENMP
        double end_time = omp_get_wtime();
        #else
        double end_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        #endif
        double duration = end_time - timings[name + "_start"];
        timings[name] += duration;
        counts[name]++;
    }
    
    double getTime(const std::string& name) {
        return timings[name];
    }
    
    void report() const {
        std::cout << "\nPerformance Report:\n";
        for (const auto& t : timings) {
            if (counts.find(t.first) != counts.end() && counts.at(t.first) > 0) {
                std::cout << "  " << t.first << ": " << t.second 
                          << "s (count: " << counts.at(t.first) 
                          << ", avg: " << t.second / counts.at(t.first) << "s)\n";
            }
        }
    }
};

// =============================================================================
// MME Builder
// =============================================================================

struct ThreadLocalAccumulator {
    std::vector<Triplet<double>> triplets;
    std::vector<Triplet<double>> xx_triplets;
    std::vector<Triplet<double>> xz_triplets; 
    std::vector<Triplet<double>> zz_triplets;
    
    ThreadLocalAccumulator() {
        triplets.reserve(100000);
        xx_triplets.reserve(25000);
        xz_triplets.reserve(25000);
        zz_triplets.reserve(50000);
    }
};

struct PrecomputedData {
    std::vector<std::vector<double>> inv_residual_covariance_cache;
    std::vector<std::vector<long>> effect_addresses;
    std::vector<std::vector<double>> effect_scales;
    std::vector<int> record_pattern_ids;
    std::vector<bool> is_fixed_address;
    std::vector<int> fixed_to_compressed;
    std::vector<int> random_to_compressed;
    std::vector<std::vector<double>> all_records;
    int n_records;
    int n_variables;
};

class FastMMEBuilder {
public:
    typedef SparseMatrix<double, ColMajor> SpMat;
    typedef Triplet<double> Tpl;
    
    struct MMEResult {
        SpMat XtX;  
        SpMat XtZ;  
        SpMat ZtZ_Ginv;  
        int n_fixed;
        int n_random;
        
        SpMat buildFullMatrix() const {
            std::vector<Triplet<double>> triplets;
            triplets.reserve(XtX.nonZeros() + 2*XtZ.nonZeros() + ZtZ_Ginv.nonZeros());
            
            for (int k = 0; k < XtX.outerSize(); ++k) {
                for (SpMat::InnerIterator it(XtX, k); it; ++it) {
                    triplets.emplace_back(it.row(), it.col(), it.value());
                }
            }
        
            for (int k = 0; k < XtZ.outerSize(); ++k) {
                for (SpMat::InnerIterator it(XtZ, k); it; ++it) {
                    triplets.emplace_back(it.row(), n_fixed + it.col(), it.value());
                    triplets.emplace_back(n_fixed + it.col(), it.row(), it.value());
                }
            }
        
            for (int k = 0; k < ZtZ_Ginv.outerSize(); ++k) {
                for (SpMat::InnerIterator it(ZtZ_Ginv, k); it; ++it) {
                    triplets.emplace_back(n_fixed + it.row(), n_fixed + it.col(), it.value());
                }
            }
        
            SpMat A(n_fixed + n_random, n_fixed + n_random);
            A.setFromTriplets(triplets.begin(), triplets.end());
            A.makeCompressed();
            return A;
        }
    };

    static MMEResult buildMME(bool verbose = true) {
        auto build_start = std::chrono::high_resolution_clock::now();
        
        if (verbose) std::cout << "[MME Builder] Starting construction...\n";
        
        auto params = readParametersFast();
        auto precomp = precomputeDataStructures(params);

        // Fix invalid residual covariance elements
        for (int p = 0; p < params.num_trait_miss_pattern; p++) {
            for (int i = 0; i < params.num_traits; i++) {
                for (int j = 0; j < params.num_traits; j++) {
                    int idx = p * params.num_traits * params.num_traits + i * params.num_traits + j;
                    if (abs(params.inv_residual_covariance[idx]) < 1e-12) {
                        if (i == j) {
                            params.inv_residual_covariance[idx] = 1.0;
                        }
                    }
                }
            }
        }
        
        auto triplets = processDataParallel(params, precomp);
        auto result = assembleSparseMatrices(triplets, precomp, params, verbose);
        
        auto build_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double>(build_end - build_start).count();
        
        if (verbose) {
            std::cout << "[MME Builder] Total time: " << (total_time*1000) << "ms\n";
            std::cout << "  Matrix size: " << (result.n_fixed + result.n_random) << "x" << (result.n_fixed + result.n_random) << "\n";
        }
        
        return result;
    }

public:
    struct Parameters {
        int num_variables, num_traits, num_total_effects;
        int num_trait_miss_pattern, num_random_matrix;
        std::vector<int> trait_columns;
        std::vector<int> model_effect_exist;
        std::vector<int> weight_columns;
        std::vector<int> rank_cov;
        std::vector<long> effect;
        long max_address, num_tiny_effect_trait_block;
        double missing, diag_blockM;
        std::vector<double> inv_residual_covariance;
        std::vector<std::vector<double>> inv_covariance;
        std::string file_data;
        std::vector<std::string> trait_miss_str;
        std::vector<std::string> file_inverse;
        std::vector<long> block_address_sum;
        std::vector<long> miss_address_sum;
        long full_max_address;
    };
    
    static Parameters readParametersFast() {
        Parameters params;
        
        std::ifstream param_file("blockM.par", std::ios::binary);
        if (!param_file.is_open()) {
            throw std::runtime_error("Cannot open blockM.par");
        }
        
        param_file.read((char*)&params.num_variables, sizeof(int));
        param_file.read((char*)&params.num_traits, sizeof(int));
        param_file.read((char*)&params.num_total_effects, sizeof(int));
        param_file.read((char*)&params.num_trait_miss_pattern, sizeof(int));
        param_file.read((char*)&params.num_random_matrix, sizeof(int));
        
        params.trait_columns.resize(params.num_traits);
        params.model_effect_exist.resize(params.num_total_effects * params.num_traits);
        params.weight_columns.resize(params.num_traits);
        params.rank_cov.resize(params.num_random_matrix);
        params.effect.resize(11 * params.num_total_effects);
        params.inv_residual_covariance.resize(params.num_trait_miss_pattern * params.num_traits * params.num_traits);
        
        param_file.read((char*)params.trait_columns.data(), params.num_traits * sizeof(int));
        param_file.read((char*)params.model_effect_exist.data(), params.num_total_effects * params.num_traits * sizeof(int));
        param_file.read((char*)params.weight_columns.data(), params.num_traits * sizeof(int));
        param_file.read((char*)params.rank_cov.data(), params.num_random_matrix * sizeof(int));
        param_file.read((char*)params.effect.data(), 11 * params.num_total_effects * sizeof(long));
        param_file.read((char*)&params.max_address, sizeof(long));
        param_file.read((char*)&params.num_tiny_effect_trait_block, sizeof(long));
        param_file.read((char*)&params.missing, sizeof(double));
        param_file.read((char*)&params.diag_blockM, sizeof(double));
        param_file.read((char*)params.inv_residual_covariance.data(), 
                       params.num_trait_miss_pattern * params.num_traits * params.num_traits * sizeof(double));
        
        params.inv_covariance.resize(params.num_random_matrix);
        for (int j = 0; j < params.num_random_matrix; j++) {
            params.inv_covariance[j].resize(params.rank_cov[j] * params.rank_cov[j]);
            param_file.read((char*)params.inv_covariance[j].data(), params.rank_cov[j] * params.rank_cov[j] * sizeof(double));
        }
        
        char file_data[FNAME_LEN];
        param_file.read(file_data, FNAME_LEN);
        params.file_data = std::string(file_data);

        params.trait_miss_str.resize(params.num_trait_miss_pattern);
        for (int j = 0; j < params.num_trait_miss_pattern; j++) {
            char temp[params.num_traits + 1];
            param_file.read(temp, params.num_traits + 1);
            temp[params.num_traits] = '\0';
            params.trait_miss_str[j] = std::string(temp, params.num_traits);
        }

        bool need_fix_patterns = false;
        for (int j = 0; j < params.num_trait_miss_pattern; j++) {
            if (params.trait_miss_str[j].empty() || 
                params.trait_miss_str[j].find_first_not_of('\0') == std::string::npos) {
                need_fix_patterns = true;
                break;
            }
        }

        if (need_fix_patterns) {
            auto original_inv_residual_cov = params.inv_residual_covariance;
            params.trait_miss_str.clear();
    
            if (params.num_traits == 2) {
                params.trait_miss_str = {"01", "10", "11"};
                params.num_trait_miss_pattern = 3;
            } else {
                params.trait_miss_str = {std::string(params.num_traits, '1')};
                params.num_trait_miss_pattern = 1;
            }
    
            params.inv_residual_covariance.resize(params.num_trait_miss_pattern * params.num_traits * params.num_traits, 0.0);
            for (int p = 0; p < params.num_trait_miss_pattern; p++) {
                for (int i = 0; i < params.num_traits; i++) {
                    for (int j = 0; j < params.num_traits; j++) {
                        params.inv_residual_covariance[p * params.num_traits * params.num_traits + i * params.num_traits + j] = 
                            original_inv_residual_cov[i * params.num_traits + j];
                    }
                }
            }
        }
        
        params.file_inverse.resize(params.num_random_matrix);
        for (int j = 0; j < params.num_random_matrix; j++) {
            char temp[FNAME_LEN];
            param_file.read(temp, FNAME_LEN);
            temp[FNAME_LEN-1] = '\0';
            std::string file_path(temp);
            size_t null_pos = file_path.find('\0');
            if (null_pos != std::string::npos) {
                file_path = file_path.substr(0, null_pos);
            }
            params.file_inverse[j] = file_path;
        }
        param_file.close();

        for (int j = 0; j < params.num_random_matrix; j++) {
            if (params.file_inverse[j].empty() || params.file_inverse[j].find_first_not_of('\0') == std::string::npos) {
                for (int k = 0; k < params.num_total_effects; k++) {
                    if (params.effect[k * 11 + 3] == j + 1) {
                        long matrix_type = params.effect[k * 11 + 10];
                        if (matrix_type == 1) {
                            params.file_inverse[j] = "inva";
                        } else if (matrix_type == 2) {
                            params.file_inverse[j] = "invg";
                        } else if (matrix_type == 3) {
                            params.file_inverse[j] = "invh";
                        } else if (matrix_type == 4) {
                            params.file_inverse[j] = "piblup-inv_f" + std::to_string(j) + "_recode";
                        } else {
                            params.file_inverse[j] = "inva";
                        }
                        break;
                    }
                }
            }
        }

        bool pattern_1_exists = false;
        for (int i = 0; i < params.num_trait_miss_pattern; i++) {
            if (params.trait_miss_str[i] == "1") {
                pattern_1_exists = true;
                break;
            }
        }
        
        if (!pattern_1_exists && params.num_traits == 1) {
            params.trait_miss_str.push_back("1");
            params.num_trait_miss_pattern++;
            params.inv_residual_covariance.resize(params.num_trait_miss_pattern * params.num_traits * params.num_traits);
            for (int i = 0; i < params.num_traits; i++) {
                for (int j = 0; j < params.num_traits; j++) {
                    params.inv_residual_covariance[(params.num_trait_miss_pattern-1) * params.num_traits * params.num_traits + i * params.num_traits + j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }

        int num_effect_block = 0;
        for (long i = params.num_total_effects - 1; i >= 0; i--) {
            if (params.effect[i * 11 + 7] > num_effect_block) {
                num_effect_block = params.effect[i * 11 + 7];
            }
        }
        
        params.block_address_sum.resize(num_effect_block, 0);
        for (long i = 0; i < num_effect_block; i++) {
            for (long j = 0; j < params.num_total_effects; j++) {
                if (params.effect[j * 11 + 7] >= i + 1) {
                    break;
                }
                params.block_address_sum[i] += params.effect[j * 11 + 6] * params.num_traits;
            }
        }

        params.full_max_address = 
            params.block_address_sum[params.effect[(params.num_total_effects - 1) * 11 + 7] - 1] +
            (params.effect[(params.num_total_effects - 1) * 11 + 6] - 1) * params.effect[(params.num_total_effects - 1) * 11 + 8] * params.num_traits +
            (params.effect[(params.num_total_effects - 1) * 11 + 9] - 1) * params.num_traits +
            params.num_traits;

        std::ifstream miss_file("null.add.sum", std::ios::binary);
        if (!miss_file.is_open()) {
            throw std::runtime_error("Cannot open null.add.sum");
        }
        params.miss_address_sum.resize(params.full_max_address);
        miss_file.read((char*)params.miss_address_sum.data(), params.full_max_address * sizeof(long));
        miss_file.close();
        
        return params;
    }
    
    static PrecomputedData precomputeDataStructures(const Parameters& params) {
        PrecomputedData precomp;
        
        precomp.is_fixed_address.resize(params.max_address, false);
        
        for (int j = 0; j < params.num_total_effects; j++) {
            if (params.effect[j * 11 + 3] == 0) {
                long base_addr = params.block_address_sum[params.effect[j * 11 + 7] - 1] +
                                (params.effect[j * 11 + 9] - 1) * params.num_traits;
                long levels = params.effect[j * 11 + 6];
                long stride = params.effect[j * 11 + 8];
                
                for (long level = 0; level < levels; level++) {
                    for (int trait = 0; trait < params.num_traits; trait++) {
                        long addr = base_addr + level * stride * params.num_traits + trait;
                        if (addr >= 0 && addr < params.full_max_address) {
                            addr -= params.miss_address_sum[addr];
                            if (addr >= 0 && addr < params.max_address) {
                                precomp.is_fixed_address[addr] = true;
                            }
                        }
                    }
                }
            }
        }
        
        precomp.fixed_to_compressed.resize(params.max_address, -1);
        precomp.random_to_compressed.resize(params.max_address, -1);
        int fixed_idx = 0, random_idx = 0;
        
        for (int i = 0; i < params.max_address; i++) {
            if (precomp.is_fixed_address[i]) {
                precomp.fixed_to_compressed[i] = fixed_idx++;
            } else {
                precomp.random_to_compressed[i] = random_idx++;
            }
        }
        
        std::ifstream data_file(params.file_data, std::ios::binary);
        if (!data_file.is_open()) {
            throw std::runtime_error("Cannot open data file: " + params.file_data);
        }
        
        data_file.seekg(0, std::ios::end);
        size_t file_size = data_file.tellg();
        data_file.seekg(0, std::ios::beg);
        
        precomp.n_records = file_size / (params.num_variables * sizeof(double));
        precomp.n_variables = params.num_variables;
        
        std::vector<double> all_data(precomp.n_records * params.num_variables);
        data_file.read((char*)all_data.data(), file_size);
        data_file.close();
        
        precomp.all_records.resize(precomp.n_records);
        #pragma omp parallel for
        for (int i = 0; i < precomp.n_records; i++) {
            precomp.all_records[i].resize(params.num_variables);
            std::memcpy(precomp.all_records[i].data(), 
                       &all_data[i * params.num_variables], 
                       params.num_variables * sizeof(double));
        }
        
        precomp.record_pattern_ids.resize(precomp.n_records);
        #pragma omp parallel for
        for (int rec = 0; rec < precomp.n_records; rec++) {
            std::string trait_miss_pattern(params.num_traits, '0');
            for (int k = 0; k < params.num_traits; k++) {
                double trait_value = precomp.all_records[rec][params.trait_columns[k] - 1];
                trait_miss_pattern[k] = (trait_value == params.missing) ? '0' : '1';
            }
            
            int pattern_id = -1;
            for (int k = 0; k < params.num_trait_miss_pattern; k++) {
                if (trait_miss_pattern == params.trait_miss_str[k]) {
                    pattern_id = k;
                    break;
                }
            }
            precomp.record_pattern_ids[rec] = pattern_id;
        }
        
        return precomp;
    }

    static std::vector<ThreadLocalAccumulator> processDataParallel(const Parameters& params, const PrecomputedData& precomp) {
        int num_threads = omp_get_max_threads();
        std::vector<ThreadLocalAccumulator> thread_accums(num_threads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_accum = thread_accums[tid];
        
            #pragma omp for schedule(dynamic, 1000)
            for (int rec = 0; rec < precomp.n_records; rec++) {
                const auto& record = precomp.all_records[rec];
                int trait_miss_pattern_id = precomp.record_pattern_ids[rec];
                if (trait_miss_pattern_id == -1) continue;
            
                for (int i = 0; i < params.num_traits; i++) {
                    if (record[params.trait_columns[i] - 1] == params.missing) continue;
                
                    for (int j = 0; j < params.num_total_effects; j++) {
                        if (params.model_effect_exist[i * params.num_total_effects + j] == 0) continue;
                    
                        long effect_level_j = params.effect[j * 11 + 5] ? 
                                              (long)record[params.effect[j * 11 + 5] - 1] - 1 : 0;
                        
                        long add1 = params.block_address_sum[params.effect[j * 11 + 7] - 1] +
                                   effect_level_j * params.effect[j * 11 + 8] * params.num_traits +
                                   (params.effect[j * 11 + 9] - 1) * params.num_traits + i;
                    
                        if (add1 >= 0 && add1 < params.full_max_address) {
                            add1 -= params.miss_address_sum[add1];
                        }
                        if (add1 < 0 || add1 >= params.max_address) continue;
                    
                        double s1 = (params.effect[j * 11 + 1] == 1 || params.effect[j * 11 + 1] == 3) ? 
                                    1.0 : record[params.effect[j * 11 + 0] - 1];
                    
                        for (int m = 0; m < params.num_traits; m++) {
                            if (record[params.trait_columns[m] - 1] == params.missing) continue;
                        
                            double inv_res_cov = params.inv_residual_covariance[
                                trait_miss_pattern_id * (params.num_traits * params.num_traits) + 
                                i * params.num_traits + m];
                        
                            if (params.weight_columns[i] != 0) {
                                inv_res_cov *= sqrt(record[params.weight_columns[i] - 1]);
                            }
                            if (params.weight_columns[m] != 0) {
                                inv_res_cov *= sqrt(record[params.weight_columns[m] - 1]);
                            }
                        
                            for (int n = 0; n < params.num_total_effects; n++) {
                                if (params.model_effect_exist[m * params.num_total_effects + n] == 0) continue;

                                bool both_random = (params.effect[j * 11 + 3] != 0) && (params.effect[n * 11 + 3] != 0);
                                if (both_random && params.effect[n * 11 + 7] != params.effect[j * 11 + 7]) {
                                    continue;
                                }
                            
                                long effect_level_n = params.effect[n * 11 + 5] ? 
                                                      (long)record[params.effect[n * 11 + 5] - 1] - 1 : 0;
                            
                                long add2 = params.block_address_sum[params.effect[n * 11 + 7] - 1] +
                                           effect_level_n * params.effect[n * 11 + 8] * params.num_traits +
                                           (params.effect[n * 11 + 9] - 1) * params.num_traits + m;
                                
                                if (add2 >= 0 && add2 < params.full_max_address) {
                                    add2 -= params.miss_address_sum[add2];
                                }
                                if (add2 < 0 || add2 >= params.max_address) continue;
                            
                                double s2 = (params.effect[n * 11 + 1] == 1 || params.effect[n * 11 + 1] == 3) ?
                                           1.0 : record[params.effect[n * 11] - 1];
                            
                                double residual_var = 1.0 / params.inv_residual_covariance[0]; 
                                double val = inv_res_cov * s1 * s2 * residual_var;
                            
                                bool add1_fixed = precomp.is_fixed_address[add1];
                                bool add2_fixed = precomp.is_fixed_address[add2];
                            
                                if (add1_fixed && add2_fixed) {
                                    int compressed_add1 = precomp.fixed_to_compressed[add1];
                                    int compressed_add2 = precomp.fixed_to_compressed[add2];
                                    if (compressed_add1 >= 0 && compressed_add2 >= 0) {
                                        local_accum.xx_triplets.emplace_back(compressed_add1, compressed_add2, val);
                                        if (compressed_add1 != compressed_add2) {
                                            local_accum.xx_triplets.emplace_back(compressed_add2, compressed_add1, val);
                                        }
                                    }
                                } else if (add1_fixed && !add2_fixed) {
                                    int compressed_add1 = precomp.fixed_to_compressed[add1];
                                    int compressed_add2 = precomp.random_to_compressed[add2];
                                    if (compressed_add1 >= 0 && compressed_add2 >= 0) {
                                        local_accum.xz_triplets.emplace_back(compressed_add1, compressed_add2, val);
                                    }
                                } else if (!add1_fixed && add2_fixed) {
                                    int compressed_add1 = precomp.fixed_to_compressed[add2];
                                    int compressed_add2 = precomp.random_to_compressed[add1];
                                    if (compressed_add1 >= 0 && compressed_add2 >= 0) {
                                        local_accum.xz_triplets.emplace_back(compressed_add1, compressed_add2, val);
                                    }
                                } else {
                                    int compressed_add1 = precomp.random_to_compressed[add1];
                                    int compressed_add2 = precomp.random_to_compressed[add2];
                                    if (compressed_add1 >= 0 && compressed_add2 >= 0) {
                                        local_accum.zz_triplets.emplace_back(compressed_add1, compressed_add2, val);
                                        if (compressed_add1 != compressed_add2) {
                                            local_accum.zz_triplets.emplace_back(compressed_add2, compressed_add1, val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return thread_accums;
    }

    static MMEResult assembleSparseMatrices(const std::vector<ThreadLocalAccumulator>& thread_accums, 
                                          const PrecomputedData& precomp,
                                          const Parameters& params, bool verbose) {
        MMEResult result;
        
        result.n_fixed = std::count(precomp.is_fixed_address.begin(), precomp.is_fixed_address.end(), true);
        result.n_random = params.max_address - result.n_fixed;
        
        std::vector<Triplet<double>> xx_triplets, xz_triplets, zz_triplets;
        
        size_t xx_size = 0, zz_size = 0;
        for (const auto& accum : thread_accums) {
            xx_size += accum.xx_triplets.size();
            zz_size += accum.zz_triplets.size();
        }
        
        xx_triplets.reserve(xx_size);
        zz_triplets.reserve(zz_size);
        
        for (const auto& accum : thread_accums) {
            xx_triplets.insert(xx_triplets.end(), accum.xx_triplets.begin(), accum.xx_triplets.end());
            zz_triplets.insert(zz_triplets.end(), accum.zz_triplets.begin(), accum.zz_triplets.end());
        }

        // Compute XtZ directly
        for (int rec = 0; rec < precomp.n_records; rec++) {
            const auto& record = precomp.all_records[rec];
            int trait_miss_pattern_id = precomp.record_pattern_ids[rec];
            if (trait_miss_pattern_id == -1) continue;
            
            for (int i = 0; i < params.num_traits; i++) {
                if (record[params.trait_columns[i] - 1] == params.missing) continue;
                
                for (int j = 0; j < params.num_total_effects; j++) {
                    if (params.effect[j * 11 + 3] != 0) continue;
                    if (params.model_effect_exist[i * params.num_total_effects + j] == 0) continue;
                    
                    long effect_level_j = params.effect[j * 11 + 5] ? 
                                          (long)record[params.effect[j * 11 + 5] - 1] - 1 : 0;
                    
                    long fixed_addr = params.block_address_sum[params.effect[j * 11 + 7] - 1] +
                                     effect_level_j * params.effect[j * 11 + 8] * params.num_traits +
                                     (params.effect[j * 11 + 9] - 1) * params.num_traits + i;
                    
                    if (fixed_addr >= 0 && fixed_addr < params.full_max_address) {
                        fixed_addr -= params.miss_address_sum[fixed_addr];
                    }
                    if (fixed_addr < 0 || fixed_addr >= params.max_address) continue;
                    if (!precomp.is_fixed_address[fixed_addr]) continue;
                    
                    double s1 = (params.effect[j * 11 + 1] == 1 || params.effect[j * 11 + 1] == 3) ? 
                               1.0 : record[params.effect[j * 11 + 0] - 1];
                    
                    for (int m = 0; m < params.num_traits; m++) {
                        if (record[params.trait_columns[m] - 1] == params.missing) continue;
                        
                        double inv_res_cov = params.inv_residual_covariance[
                            trait_miss_pattern_id * (params.num_traits * params.num_traits) + 
                            i * params.num_traits + m];
                        
                        if (params.weight_columns[i] != 0) {
                            inv_res_cov *= sqrt(record[params.weight_columns[i] - 1]);
                        }
                        if (params.weight_columns[m] != 0) {
                            inv_res_cov *= sqrt(record[params.weight_columns[m] - 1]);
                        }
                        
                        for (int n = 0; n < params.num_total_effects; n++) {
                            if (params.effect[n * 11 + 3] == 0) continue;
                            if (params.model_effect_exist[m * params.num_total_effects + n] == 0) continue;
                            
                            long effect_level_n = params.effect[n * 11 + 5] ? 
                                                 (long)record[params.effect[n * 11 + 5] - 1] - 1 : 0;
                            
                            long random_addr = params.block_address_sum[params.effect[n * 11 + 7] - 1] +
                                              effect_level_n * params.effect[n * 11 + 8] * params.num_traits +
                                              (params.effect[n * 11 + 9] - 1) * params.num_traits + m;
                            
                            if (random_addr >= 0 && random_addr < params.full_max_address) {
                                random_addr -= params.miss_address_sum[random_addr];
                            }
                            if (random_addr < 0 || random_addr >= params.max_address) continue;
                            if (precomp.is_fixed_address[random_addr]) continue;
                            
                            double s2 = (params.effect[n * 11 + 1] == 1 || params.effect[n * 11 + 1] == 3) ?
                                       1.0 : record[params.effect[n * 11] - 1];
                            
                            double residual_var = 1.0 / params.inv_residual_covariance[0]; 
                            double val = inv_res_cov * s1 * s2 * residual_var;
                            
                            int compressed_fixed = precomp.fixed_to_compressed[fixed_addr];
                            int compressed_random = precomp.random_to_compressed[random_addr];
                            
                            if (compressed_fixed >= 0 && compressed_random >= 0) {
                                xz_triplets.emplace_back(compressed_fixed, compressed_random, val);
                            }
                        }
                    }
                }
            }
        }

        // G-inverse processing
        for (int i = 0; i < params.num_random_matrix; i++) {
            int effect_index = -1;
            long effect_level = 0;
            int random_mat_type = 0;

            for (int j = 0; j < params.num_total_effects; j++) {
                if (params.effect[j * 11 + 3] == i + 1) {
                    effect_index = j;
                    effect_level = params.effect[j * 11 + 6];
                    random_mat_type = params.effect[j * 11 + 10];
                    break;
                }
            }
            
            if (effect_index < 0) continue;
            
            long base_addr = params.block_address_sum[params.effect[effect_index * 11 + 7] - 1] +
                            (params.effect[effect_index * 11 + 9] - 1) * params.num_traits;

            if (random_mat_type == 0) {
                // Permanent environmental effect: identity matrix
                for (long level = 0; level < effect_level; level++) {
                    for (int trait_i = 0; trait_i < params.num_traits; trait_i++) {
                        for (int trait_j = trait_i; trait_j < params.num_traits; trait_j++) {
                            long final_row_addr = base_addr + level * params.num_traits + trait_i;
                            long final_col_addr = base_addr + level * params.num_traits + trait_j;
                            
                            if (final_row_addr < params.full_max_address) {
                                final_row_addr -= params.miss_address_sum[final_row_addr];
                            }
                            if (final_col_addr < params.full_max_address) {
                                final_col_addr -= params.miss_address_sum[final_col_addr];
                            }
                            
                            if (final_row_addr >= 0 && final_row_addr < params.max_address &&
                                final_col_addr >= 0 && final_col_addr < params.max_address &&
                                !precomp.is_fixed_address[final_row_addr] && 
                                !precomp.is_fixed_address[final_col_addr]) {
                                
                                int compressed_row = precomp.random_to_compressed[final_row_addr];
                                int compressed_col = precomp.random_to_compressed[final_col_addr];
                                
                                if (compressed_row >= 0 && compressed_col >= 0) {
                                    double cov_inv_elem = params.inv_covariance[i][trait_i * params.rank_cov[i] + trait_j];
                                    double residual_var;
                                    if (params.num_traits == 1) {
                                        residual_var = 1.0 / params.inv_residual_covariance[0];
                                    } else {
                                        double inv_var = params.inv_residual_covariance[trait_i * params.num_traits + trait_j];
                                        if (abs(inv_var) < 1e-12) continue;
                                        residual_var = 1.0 / inv_var;
                                    }
                                    
                                    double final_val = cov_inv_elem * residual_var;
                                    
                                    if (abs(final_val) > 1e-12) {
                                        zz_triplets.emplace_back(compressed_row, compressed_col, final_val);
                                        if (compressed_row != compressed_col) {
                                            zz_triplets.emplace_back(compressed_col, compressed_row, final_val);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                continue;
            }
            
            // Read G-inverse file
            std::ifstream ginv_file(params.file_inverse[i], std::ios::binary);
            
            if (ginv_file.is_open()) {
                long row, col;
                double val;
                
                while (ginv_file.read((char*)&row, sizeof(long)) &&
                       ginv_file.read((char*)&col, sizeof(long)) &&
                       ginv_file.read((char*)&val, sizeof(double))) {
                    
                    row--; col--;
                    
                    if (row <= col && row >= 0 && row < effect_level && col >= 0 && col < effect_level) {
                        bool is_random_regression = (params.rank_cov[i] > params.num_traits);
                        int matrix_size = is_random_regression ? params.rank_cov[i] : params.num_traits;

                        for (int coeff_i = 0; coeff_i < matrix_size; coeff_i++) {
                            for (int coeff_j = 0; coeff_j < matrix_size; coeff_j++) {
                                long final_row_addr, final_col_addr;
                                if (is_random_regression) {
                                    final_row_addr = base_addr + row * params.rank_cov[i] + coeff_i;
                                    final_col_addr = base_addr + col * params.rank_cov[i] + coeff_j;
                                } else {
                                    final_row_addr = base_addr + row * params.num_traits + coeff_i;
                                    final_col_addr = base_addr + col * params.num_traits + coeff_j;
                                }
                                
                                if (final_row_addr < params.full_max_address) {
                                    final_row_addr -= params.miss_address_sum[final_row_addr];
                                }
                                if (final_col_addr < params.full_max_address) {
                                    final_col_addr -= params.miss_address_sum[final_col_addr];
                                }
                                
                                if (final_row_addr >= 0 && final_row_addr < params.max_address &&
                                    final_col_addr >= 0 && final_col_addr < params.max_address &&
                                    !precomp.is_fixed_address[final_row_addr] && 
                                    !precomp.is_fixed_address[final_col_addr]) {
                                    
                                    int compressed_row = precomp.random_to_compressed[final_row_addr];
                                    int compressed_col = precomp.random_to_compressed[final_col_addr];
                                    
                                    if (compressed_row >= 0 && compressed_col >= 0 && 
                                        compressed_row < result.n_random && compressed_col < result.n_random) {
                                        
                                        double cov_inv_elem = params.inv_covariance[i][coeff_i * params.rank_cov[i] + coeff_j];
                                        double residual_var;
                                        if (params.num_traits == 1) {
                                            residual_var = 1.0 / params.inv_residual_covariance[0];
                                        } else {
                                            double inv_var = params.inv_residual_covariance[coeff_i * params.num_traits + coeff_j];
                                            if (abs(inv_var) < 1e-12) continue;
                                            residual_var = 1.0 / inv_var;
                                        }
                                        
                                        double kronecker_val = val * cov_inv_elem * residual_var;
                                        
                                        if (abs(kronecker_val) > 1e-12) {
                                            zz_triplets.emplace_back(compressed_row, compressed_col, kronecker_val);
                                            if (compressed_row != compressed_col) {
                                                zz_triplets.emplace_back(compressed_col, compressed_row, kronecker_val);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                ginv_file.close();
            } else {
                // Use identity matrix if G-inverse file not found
                for (long level = 0; level < effect_level; level++) {
                    for (int trait_i = 0; trait_i < params.num_traits; trait_i++) {
                        for (int trait_j = 0; trait_j < params.num_traits; trait_j++) {
                            double cov_inv_elem = params.inv_covariance[i][trait_i * params.rank_cov[i] + trait_j];
                            
                            long final_addr = base_addr + level * params.num_traits + trait_i;
                            if (final_addr < params.full_max_address) {
                                final_addr -= params.miss_address_sum[final_addr];
                            }
                            
                            if (final_addr >= 0 && final_addr < params.max_address &&
                                !precomp.is_fixed_address[final_addr]) {
                                
                                int compressed_idx = precomp.random_to_compressed[final_addr];
                                if (compressed_idx >= 0 && compressed_idx < result.n_random) {
                                    if (trait_i == trait_j && abs(cov_inv_elem) > 1e-12) {
                                        zz_triplets.emplace_back(compressed_idx, compressed_idx, cov_inv_elem);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
// Build sparse matrices
        result.XtX.resize(result.n_fixed, result.n_fixed);
        result.XtX.setFromTriplets(xx_triplets.begin(), xx_triplets.end());
        result.XtX.makeCompressed();
        
        result.XtZ.resize(result.n_fixed, result.n_random);
        result.XtZ.setFromTriplets(xz_triplets.begin(), xz_triplets.end());
        result.XtZ.makeCompressed();
        
        result.ZtZ_Ginv.resize(result.n_random, result.n_random);
        result.ZtZ_Ginv.setFromTriplets(zz_triplets.begin(), zz_triplets.end());
        result.ZtZ_Ginv.makeCompressed();
        
        if (verbose) {
            std::cout << "  XtX: " << result.XtX.rows() << "x" << result.XtX.cols() 
                      << " (" << result.XtX.nonZeros() << " non-zeros)\n";
            std::cout << "  XtZ: " << result.XtZ.rows() << "x" << result.XtZ.cols() 
                      << " (" << result.XtZ.nonZeros() << " non-zeros)\n";
            std::cout << "  ZtZ+G^-1: " << result.ZtZ_Ginv.rows() << "x" << result.ZtZ_Ginv.cols() 
                      << " (" << result.ZtZ_Ginv.nonZeros() << " non-zeros)\n";
        }
        
        return result;
    }
};

// =============================================================================
// Block Diagonal Preconditioner
// =============================================================================
class BlockDiagonalPreconditioner {
private:
    typedef SparseMatrix<double, ColMajor> SpMat;
    typedef Eigen::Matrix<double, Dynamic, Dynamic, ColMajor> MatrixXd;
    
    int n_fixed;
    int n_random;
    int total_size;
    
    SimplicialLDLT<SpMat> chol_ff;
    SimplicialLDLT<SpMat> chol_rr;
    
    AlignedVector<double> diag_ff;
    AlignedVector<double> diag_rr;
    
public:
    BlockDiagonalPreconditioner(int nf, int nr) 
        : n_fixed(nf), n_random(nr), total_size(nf + nr) {
    }
    
    void build(const SpMat& XtX, const SpMat& XtZ, const SpMat& ZtZ_Ginv, double regularization = 0) {
        diag_ff.resize(n_fixed);
        diag_rr.resize(n_random);
        
        // Build fixed effects block
        SpMat XtX_reg = XtX;
        for (int i = 0; i < n_fixed; i++) {
            XtX_reg.coeffRef(i, i) += regularization;
        }
        chol_ff.compute(XtX_reg);
        // 在这之后添加
        if (chol_ff.info() != Success) {
            std::cerr << "WARNING: Fixed effects Cholesky failed!\n";
        }

        
        #pragma omp parallel for
        for (int i = 0; i < n_fixed; i++) {
            diag_ff[i] = XtX_reg.coeff(i, i);
            if (diag_ff[i] < regularization) diag_ff[i] = regularization;
        }
        
        // Build random effects block
        SpMat ZtZ_reg = ZtZ_Ginv;
        for (int i = 0; i < n_random; i++) {
            ZtZ_reg.coeffRef(i, i) += regularization;
        }
        chol_rr.compute(ZtZ_reg);

        if (chol_rr.info() != Success) {
            std::cerr << "WARNING: Random effects Cholesky failed!\n";
        }       
 
        #pragma omp parallel for
        for (int i = 0; i < n_random; i++) {
            diag_rr[i] = ZtZ_reg.coeff(i, i);
            if (diag_rr[i] < regularization) diag_rr[i] = regularization;
        }
    }
    
    void apply(const double* r, double* z) const {
        const double* r_f = r;
        const double* r_r = r + n_fixed;
        double* z_f = z;
        double* z_r = z + n_fixed;
        
        if (chol_ff.info() == Success) {
            Map<const VectorXd> r_map(r_f, n_fixed);
            Map<VectorXd> z_map(z_f, n_fixed);
            z_map = chol_ff.solve(r_map);
        } else {
            #pragma omp parallel for
            for (int i = 0; i < n_fixed; i++) {
                z_f[i] = r_f[i] / diag_ff[i];
            }
        }
        
        if (chol_rr.info() == Success) {
            Map<const VectorXd> r_map(r_r, n_random);
            Map<VectorXd> z_map(z_r, n_random);
            z_map = chol_rr.solve(r_map);
        } else {
            #pragma omp parallel for
            for (int i = 0; i < n_random; i++) {
                z_r[i] = r_r[i] / diag_rr[i];
            }
        }
    }
};

// =============================================================================
// Optimized Sparse Matrix-Vector Multiplication
// =============================================================================
void optimizedSpMV(const SparseMatrix<double, ColMajor>& A, 
                  const double* x, double* y, 
                  PerformanceMonitor& perf_mon) {
    perf_mon.start("SpMV");
    
    const int n = A.rows();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = 0.0;
    }
    
    #pragma omp parallel
    {
        int num_cols = A.outerSize();
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int cols_per_thread = (num_cols + num_threads - 1) / num_threads;
        int start_col = tid * cols_per_thread;
        int end_col = std::min(start_col + cols_per_thread, num_cols);
        
        for (int k = start_col; k < end_col; ++k) {
            for (SparseMatrix<double, ColMajor>::InnerIterator it(A, k); it; ++it) {
                #pragma omp atomic
                y[it.row()] += it.value() * x[k];
            }
        }
    }
    
    perf_mon.stop("SpMV");
}

// =============================================================================
// Block Preconditioned Conjugate Gradient Solver
// =============================================================================
class BlockPCGSolver {
private:
    typedef SparseMatrix<double, ColMajor> SpMat;
    const SpMat& A;
    BlockDiagonalPreconditioner& precond;
    PerformanceMonitor& perf_mon;
    
    AlignedVector<double> r;
    AlignedVector<double> z;
    AlignedVector<double> p;
    AlignedVector<double> Ap;
    
public:
    BlockPCGSolver(const SpMat& matrix, BlockDiagonalPreconditioner& prec, PerformanceMonitor& pm) 
        : A(matrix), precond(prec), perf_mon(pm) {
        int n = A.rows();
        r.resize(n);
        z.resize(n);
        p.resize(n);
        Ap.resize(n);
    }
    
    std::pair<int, double> solve(const double* b, double* x, 
                                  double tol = 1e-6, 
                                  int max_iter = 500,
                                  bool verbose = false) {
        perf_mon.start("PCG_solve");
        
        int n = A.rows();
        
        double b_norm = 0.0;
        #pragma omp parallel for reduction(+:b_norm)
        for (int i = 0; i < n; i++) {
            x[i] = 0.0;
            r[i] = b[i];
            b_norm += b[i] * b[i];
        }
        b_norm = sqrt(b_norm);
        
        if (b_norm < 1e-14) {
            perf_mon.stop("PCG_solve");
            return std::make_pair(0, 0.0);
        }
        
        perf_mon.start("PCG_precond");
        precond.apply(r.data(), z.data());
        perf_mon.stop("PCG_precond");
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p[i] = z[i];
        }
        
        double rz_old = 0.0;
        #pragma omp parallel for reduction(+:rz_old)
        for (int i = 0; i < n; i++) {
            rz_old += r[i] * z[i];
        }
        
        int iter = 0;
        double residual_norm = 0.0;
        
        for (iter = 0; iter < max_iter; iter++) {
            optimizedSpMV(A, p.data(), Ap.data(), perf_mon);
            
            double pAp = 0.0;
            #pragma omp parallel for reduction(+:pAp)
            for (int i = 0; i < n; i++) {
                pAp += p[i] * Ap[i];
            }
            
            if (abs(pAp) < 1e-14) break;
            
            double alpha = rz_old / pAp;
            
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
            
            residual_norm = 0.0;
            #pragma omp parallel for reduction(+:residual_norm)
            for (int i = 0; i < n; i++) {
                residual_norm += r[i] * r[i];
            }
            residual_norm = sqrt(residual_norm) / b_norm;
            
            if (residual_norm < tol) break;
            
            perf_mon.start("PCG_precond");
            precond.apply(r.data(), z.data());
            perf_mon.stop("PCG_precond");
            
            double rz_new = 0.0;
            #pragma omp parallel for reduction(+:rz_new)
            for (int i = 0; i < n; i++) {
                rz_new += r[i] * z[i];
            }
            
            if (abs(rz_new) < 1e-14) break;
            
            double beta = rz_new / rz_old;
            
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                p[i] = z[i] + beta * p[i];
            }
            
            rz_old = rz_new;
        }
        
        perf_mon.stop("PCG_solve");
        return std::make_pair(iter, residual_norm);
    }
};

// =============================================================================
// Hutchinson++ Multi-Vector Sampling
// =============================================================================
class HutchinsonPlusPlus {
private:
    int mme_size;
    int num_vectors;
    int block_size;      
    int n_fixed;       
    int n_random;      
    AlignedVector<double> Z;
    AlignedVector<double> X;
    std::vector<std::mt19937_64> rngs;
    std::vector<std::vector<double>> block_sum;
    std::vector<std::vector<double>> block_sum2;
    
public:
    HutchinsonPlusPlus(int size, int nv, int threads, 
                      int bs, int nf, int nr) 
        : mme_size(size), num_vectors(nv), 
          block_size(bs), n_fixed(nf), n_random(nr){
        Z.resize(mme_size * num_vectors);
        X.resize(mme_size * num_vectors);
        
        std::random_device rd;
        rngs.resize(threads);
        for (int i = 0; i < threads; i++) {
            rngs[i].seed(rd() + i);
        }
        // 初始化块存储
        int n_blocks, elements_per_block;
        if (block_size == 1) {
            // 标量模式
            n_blocks = mme_size;
            elements_per_block = 1;
        } else {
            // 块模式
            n_blocks = n_random / block_size;
            elements_per_block = block_size * block_size;
        }
        
        block_sum.resize(n_blocks);
        block_sum2.resize(n_blocks);
        for (int i = 0; i < n_blocks; i++) {
            block_sum[i].resize(elements_per_block, 0.0);
            block_sum2[i].resize(elements_per_block, 0.0);
        }
    }
    
    void generateRandomVectors(int sample_id, PerformanceMonitor& perf_mon) {
        perf_mon.start("random_gen");
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::uniform_int_distribution<int> rademacher(0, 1);
            
            #pragma omp for
            for (int v = 0; v < num_vectors; v++) {
                for (int i = 0; i < mme_size; ++i) {
                    Z[v * mme_size + i] = (rademacher(rngs[tid]) == 0) ? 1.0 : -1.0;
                }
            }    
        }        
        
        perf_mon.stop("random_gen");
    }
    
    void solveBatch(BlockPCGSolver& solver, double pcg_tol, int pcg_maxiter, 
               PerformanceMonitor& perf_mon) {
        perf_mon.start("batch_solve");
        
        for (int v = 0; v < num_vectors; v++) {
            solver.solve(&Z[v * mme_size], &X[v * mme_size], pcg_tol, pcg_maxiter, false);
        }
        
        perf_mon.stop("batch_solve");
    }

    void updateEstimates(PerformanceMonitor& perf_mon) {
        perf_mon.start("estimate_update");
    
        if (block_size == 1) {
            // 标量模式：原有逻辑
            #pragma omp parallel for
            for (int i = 0; i < mme_size; ++i) {
                for (int v = 0; v < num_vectors; v++) {
                    double contrib = Z[v * mme_size + i] * X[v * mme_size + i];
                    block_sum[i][0] += contrib;
                    block_sum2[i][0] += contrib * contrib;
                }
            }
        } else {
            // 块模式：新逻辑
            int n_animals = n_random / block_size;
        
            #pragma omp parallel for
            for (int animal_id = 0; animal_id < n_animals; ++animal_id) {
                for (int v = 0; v < num_vectors; v++) {
                // 计算外积 Z_i × X_i^T
                    for (int p = 0; p < block_size; p++) {
                        int idx_p = n_fixed + animal_id * block_size + p;
                        double z_p = Z[v * mme_size + idx_p];
                    
                        for (int q = 0; q < block_size; q++) {
                            int idx_q = n_fixed + animal_id * block_size + q;
                            double x_q = X[v * mme_size + idx_q];
                        
                            double contrib = z_p * x_q;
                            int flat_idx = p * block_size + q;
                        
                            block_sum[animal_id][flat_idx] += contrib;
                            block_sum2[animal_id][flat_idx] += contrib * contrib;
                        }
                    }
                }
            }
        }
    
        perf_mon.stop("estimate_update");
    } 
// 对称化（仅块模式需要）
    void symmetrize(int num_samples) {
        if (block_size == 1) {
            return;  // 标量模式不需要
        }
        
        int n_animals = block_sum.size();
        for (int animal_id = 0; animal_id < n_animals; animal_id++) {
            for (int p = 0; p < block_size; p++) {
                for (int q = p + 1; q < block_size; q++) {
                    int idx_pq = p * block_size + q;
                    int idx_qp = q * block_size + p;
                    
                    double avg = (block_sum[animal_id][idx_pq] + 
                                 block_sum[animal_id][idx_qp]) / 2.0;
                    
                    block_sum[animal_id][idx_pq] = avg;
                    block_sum[animal_id][idx_qp] = avg;
                    
                    // 方差也对称化
                    double avg2 = (block_sum2[animal_id][idx_pq] + 
                                  block_sum2[animal_id][idx_qp]) / 2.0;
                    
                    block_sum2[animal_id][idx_pq] = avg2;
                    block_sum2[animal_id][idx_qp] = avg2;
                }
            }
        }
    }
    
    // 计算最大误差（用于收敛判断）
    double getMaxError(int num_samples) const {
        if (num_samples < 10) return 1.0;
        
        double max_error = 0.0;
        int n_blocks = block_sum.size();
        
        if (block_size == 1) {
            // 标量模式：检查所有元素
            #pragma omp parallel for reduction(max:max_error)
            for (int i = 0; i < n_blocks; i++) {
                double mean = block_sum[i][0] / num_samples;
                double var = (block_sum2[i][0] - 
                             block_sum[i][0] * block_sum[i][0] / num_samples) 
                             / (num_samples - 1);
                double stderr_val = sqrt(var) / sqrt(num_samples);
                double rel_err = stderr_val / std::max(fabs(mean), 1e-8);
                
                if (rel_err > max_error) {
                    max_error = rel_err;
                }
            }
        } else {
            // 块模式：只检查对角元素
            #pragma omp parallel for reduction(max:max_error)
            for (int animal_id = 0; animal_id < n_blocks; animal_id++) {
                for (int k = 0; k < block_size; k++) {
                    int idx = k * block_size + k;  // 对角元素
                    
                    double mean = block_sum[animal_id][idx] / num_samples;
                    double var = (block_sum2[animal_id][idx] - 
                                 block_sum[animal_id][idx] * block_sum[animal_id][idx] / num_samples) 
                                 / (num_samples - 1);
                    double stderr_val = sqrt(var) / sqrt(num_samples);
                    double rel_err = stderr_val / std::max(fabs(mean), 1e-8);
                    
                    if (rel_err > max_error) {
                        max_error = rel_err;
                    }
                }
            }
        }
        
        return max_error;
    }
    
    // 输出结果
    void writeResults(const std::string& filename, int num_samples) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    if (block_size == 1) {
        // k=1：标量模式，只输出一列
        file << "# diagonal_element\n";
        
        // 输出所有位置（固定效应 + 随机效应）
        for (size_t i = 0; i < block_sum.size(); i++) {
            double mean = block_sum[i][0] / num_samples;
            
            // 修正负值
            if (mean <= 0) {
                double var = (block_sum2[i][0] - 
                             block_sum[i][0] * block_sum[i][0] / num_samples) 
                             / (num_samples - 1);
                double stderr_val = sqrt(var) / sqrt(num_samples);
                mean = stderr_val > 0 ? stderr_val : 1e-6;
            }
            
            file << std::scientific << std::setprecision(12) << mean << "\n";
        }
    } else {
        // k>1：块模式，输出k列
        file << "#";
        for (int i = 1; i <= block_size; i++) {
            file << " C_" << i;
        }
        file << "\n";
        
        // 输出固定效应部分（只有对角线元素）
        for (int i = 0; i < n_fixed; i++) {
            // 从标量对角线获取固定效应的值
            // 注意：固定效应没有存储在block_sum中，需要特殊处理
            // 第一列是对角线，其他列是0
            file << "0.0";  // 占位，实际应该从某处获取固定效应对角线
            for (int j = 1; j < block_size; j++) {
                file << " 0";
            }
            file << "\n";
        }
        
        // 输出随机效应部分（每个动物的5×5矩阵，按行展开）
        int n_animals = block_sum.size();
        for (int animal_id = 0; animal_id < n_animals; animal_id++) {
            // 输出该动物5×5矩阵的每一行
            for (int row = 0; row < block_size; row++) {
                for (int col = 0; col < block_size; col++) {
                    int idx = row * block_size + col;
                    double mean = block_sum[animal_id][idx] / num_samples;
                    
                    // 修正对角元素的负值
                    if (row == col && mean <= 0) {
                        double var = (block_sum2[animal_id][idx] - 
                                     block_sum[animal_id][idx] * block_sum[animal_id][idx] / num_samples) 
                                     / (num_samples - 1);
                        double stderr_val = sqrt(var) / sqrt(num_samples);
                        mean = stderr_val > 0 ? stderr_val : 1e-6;
                    }
                    
                    if (col > 0) file << " ";
                    file << std::scientific << std::setprecision(12) << mean;
                }
                file << "\n";
            }
        }
    }
    
    file.close();
}
    




};


class ExactBlockInverter {
public:
    typedef SparseMatrix<double, ColMajor> SpMat;
    
    // 修改函数签名，添加 precomp 参数
    static std::vector<Eigen::MatrixXd> computeExactBlocks(
        const SpMat& A,
        const std::vector<int>& animal_ids,
        int block_size,
        int n_fixed,
        const PrecomputedData& precomp,          
        const FastMMEBuilder::Parameters& params, 
        bool verbose = true
    ) {
        std::vector<Eigen::MatrixXd> results;
        
        if (verbose) {
            std::cout << "\n========================================\n";
            std::cout << "  Computing Exact Inverse Blocks\n";
            std::cout << "========================================\n";
        }
        
        auto factor_start = std::chrono::high_resolution_clock::now();
        
        SimplicialLDLT<SpMat> solver;
        solver.compute(A);
        
        if (solver.info() != Success) {
            std::cerr << "ERROR: Matrix factorization failed!\n";
            throw std::runtime_error("Cholesky decomposition failed");
        }
        
        auto factor_end = std::chrono::high_resolution_clock::now();
        double factor_time = std::chrono::duration<double>(factor_end - factor_start).count();
        
        if (verbose) {
            std::cout << "Factorization done in " << factor_time << "s\n\n";
        }
        
        auto solve_start = std::chrono::high_resolution_clock::now();
        
        // 找到第一个随机效应的信息
        int random_effect_idx = -1;
        for (int j = 0; j < params.num_total_effects; j++) {
            if (params.effect[j * 11 + 3] != 0) {  // 是随机效应
                random_effect_idx = j;
                break;
            }
        }
        
        if (random_effect_idx < 0) {
            throw std::runtime_error("No random effect found!");
        }
        
        long base_addr = params.block_address_sum[params.effect[random_effect_idx * 11 + 7] - 1];
        
        for (size_t idx = 0; idx < animal_ids.size(); idx++) {
            int animal_id = animal_ids[idx];
            
            if (verbose && (idx % 10 == 0 || idx < 5)) {
                std::cout << "Computing animal " << animal_id 
                          << " (" << (idx+1) << "/" << animal_ids.size() 
                          << ")...\n" << std::flush;
            }
            
            Eigen::MatrixXd block(block_size, block_size);
            
            // 计算这个动物每个系数的compressed索引
            std::vector<int> compressed_indices(block_size);
            for (int k = 0; k < block_size; k++) {
                // 原始地址
                long orig_addr = base_addr + animal_id * block_size + k;
                
                // 应用 miss_address_sum 调整
                if (orig_addr >= 0 && orig_addr < params.full_max_address) {
                    orig_addr -= params.miss_address_sum[orig_addr];
                }
                
                // 检查是否是随机效应
                if (orig_addr < 0 || orig_addr >= params.max_address || 
                    precomp.is_fixed_address[orig_addr]) {
                    std::cerr << "ERROR: Animal " << animal_id << " coefficient " << k 
                              << " is not a random effect!\n";
                    throw std::runtime_error("Invalid random effect address");
                }
                
                // 获取compressed索引
                int compressed_idx = precomp.random_to_compressed[orig_addr];
                if (compressed_idx < 0 || compressed_idx >= A.rows() - n_fixed) {
                    std::cerr << "ERROR: Invalid compressed index " << compressed_idx << "\n";
                    throw std::runtime_error("Invalid compressed index");
                }
                
                // 在MME矩阵中的实际位置
                compressed_indices[k] = n_fixed + compressed_idx;
                
                if (idx == 0 && verbose) {
                    std::cout << "  Coeff " << k << ": orig=" << (base_addr + animal_id * block_size + k)
                              << " -> adjusted=" << orig_addr
                              << " -> compressed=" << compressed_idx
                              << " -> MME_idx=" << compressed_indices[k] << "\n";
                }
            }
            
            // 对每个系数求解 A x = e_j
            for (int j = 0; j < block_size; j++) {
                VectorXd e_j = VectorXd::Zero(A.rows());
                e_j(compressed_indices[j]) = 1.0;  // 使用正确的compressed索引！
                
                VectorXd x = solver.solve(e_j);
                
                if (solver.info() != Success) {
                    std::cerr << "ERROR: Solve failed for animal " << animal_id 
                              << " column " << j << "\n";
                    throw std::runtime_error("Linear solve failed");
                }
                
                // 提取块的第j列
                for (int i = 0; i < block_size; i++) {
                    block(i, j) = x(compressed_indices[i]);  // 使用正确的compressed索引！
                }
            }
            
            results.push_back(block);
        }
        
        auto solve_end = std::chrono::high_resolution_clock::now();
        double solve_time = std::chrono::duration<double>(solve_end - solve_start).count();
        
        if (verbose) {
            std::cout << "\nAll blocks computed in " << solve_time << "s\n";
            std::cout << "Average time per animal: " 
                      << (solve_time / animal_ids.size()) << "s\n";
        }
        
        return results;
    }
    
   
    // 输出到文件（与Hutchinson++相同的格式）
    static void writeResults(
        const std::vector<Eigen::MatrixXd>& blocks,
        const std::vector<int>& animal_ids,
        int n_fixed,
        int block_size,
        const std::string& filename
    ) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // 输出表头
        file << "#";
        for (int i = 1; i <= block_size; i++) {
            file << " C_" << i;
        }
        file << "\n";
        
        // 输出固定效应（占位）
        for (int i = 0; i < n_fixed; i++) {
            file << "0.0";
            for (int j = 1; j < block_size; j++) {
                file << " 0";
            }
            file << "\n";
        }
        
        // 输出随机效应（按动物ID顺序）
        std::map<int, const Eigen::MatrixXd*> id_to_block;
        for (size_t i = 0; i < animal_ids.size(); i++) {
            id_to_block[animal_ids[i]] = &blocks[i];
        }
        
        // 按动物ID从0开始输出
        int max_animal_id = animal_ids.empty() ? 0 : 
                           *std::max_element(animal_ids.begin(), animal_ids.end());
        
        for (int animal_id = 0; animal_id <= max_animal_id; animal_id++) {
            if (id_to_block.find(animal_id) != id_to_block.end()) {
                // 找到了这个动物，输出其5×5矩阵的每一行
                const auto& block = *id_to_block[animal_id];
                
                for (int row = 0; row < block_size; row++) {
                    for (int col = 0; col < block_size; col++) {
                        if (col > 0) file << " ";
                        file << std::scientific << std::setprecision(12) 
                             << block(row, col);
                    }
                    file << "\n";
                }
            } else {
                // 没有计算这个动物，输出占位行
                for (int row = 0; row < block_size; row++) {
                    file << "NA";
                    for (int col = 1; col < block_size; col++) {
                        file << " NA";
                    }
                    file << "\n";
                }
            }
        }
        
        file.close();
        std::cout << "Exact results written to: " << filename << "\n";
    }
};


// =============================================================================
// MME Diagonal Solver
// =============================================================================
class MMEDiagonalSolver {
public:
    typedef SparseMatrix<double, ColMajor> SpMat;
    
    struct Config {
        int max_samples = 200;
        int threads = 0;
        double convergence_tol = 0.001;
        double pcg_tol = 1e-6;
        int pcg_maxiter = 300;
        bool verbose = true;
        std::string output_file = "mme_diagonal_results.txt";
        int block_size = 1;
        double regularization = 0.001;
        bool compute_exact = false;              
        int exact_n_animals = 10;               
        std::string exact_output = "exact_inverse.txt";  
    };

private:
    Config config;
    PerformanceMonitor perf_mon;
    
public:
    MMEDiagonalSolver() = default;
    
    void setConfig(const Config& cfg) {
        config = cfg;
        
        if (config.threads <= 0) {
            #ifdef _OPENMP
            config.threads = omp_get_max_threads();
            #else
            config.threads = 1;
            #endif
        }
        
        #ifdef _OPENMP
        omp_set_num_threads(config.threads);
        Eigen::setNbThreads(1);
        #endif
    }
    
    bool solve() {
        auto total_start = std::chrono::high_resolution_clock::now();
        std::map<std::string, double> timings;
        
        try {
            if (config.verbose) {
                std::cout << "\n========================================\n";
                std::cout << "  MME Diagonal Solver\n";
                std::cout << "========================================\n";
                std::cout << "Configuration:\n";
                std::cout << "  Threads: " << config.threads << "\n";
                std::cout << "  Max samples: " << config.max_samples << "\n";
                std::cout << "  PCG tolerance: " << config.pcg_tol << "\n";
                std::cout << "========================================\n\n";
            }
            
            // Phase 1: MME construction
            if (config.verbose) std::cout << "[1] MME construction...\n";
            auto t_build = std::chrono::high_resolution_clock::now();
            
            auto mme = FastMMEBuilder::buildMME(config.verbose);
            
            const int n_fixed = mme.n_fixed;
            const int n_random = mme.n_random;
            const int mme_size = n_fixed + n_random;
            
            timings["build_mme"] = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_build).count();

            if (config.block_size == 1) {
                if (config.verbose) {
                    std::cout << "\nMode: Scalar diagonal estimation\n";
                    std::cout << "Elements to estimate: " << mme_size << "\n";
                }
            } else {
                if (n_random % config.block_size != 0) {
                    std::cerr << "\nERROR: Random effects size (" << n_random 
                              << ") is not divisible by block_size (" << config.block_size << ")\n";
                    std::cerr << "Please check your model configuration.\n";
                    std::cerr << "Expected: n_random should be a multiple of block_size\n";
                    return false;
                }
    
                int n_animals = n_random / config.block_size;
                if (config.verbose) {
                    std::cout << "\nMode: Block diagonal estimation\n";
                    std::cout << "Block size: " << config.block_size << "×" << config.block_size << "\n";
                    std::cout << "Number of blocks: " << n_animals << "\n";
                    std::cout << "Elements per block: " << (config.block_size * config.block_size) << "\n";
                }
            }     
            // Phase 2: Block preconditioner setup
            if (config.verbose) std::cout << "\n[2] Setting up block preconditioner...\n";
            auto t_precond = std::chrono::high_resolution_clock::now();
            
            BlockDiagonalPreconditioner block_precond(n_fixed, n_random);
            block_precond.build(mme.XtX, mme.XtZ, mme.ZtZ_Ginv, config.regularization);
            
            SpMat A = mme.buildFullMatrix();
            if (config.regularization > 0) {
                for (int i = 0; i < A.rows(); i++) {
                    A.coeffRef(i, i) += config.regularization;
                }
                A.makeCompressed();
    
                if (config.verbose) {
                    std::cout << "Added regularization " << config.regularization 
                              << " to MME diagonal\n";
                }
            }

            BlockPCGSolver block_pcg(A, block_precond, perf_mon);
            
            timings["setup_precond"] = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_precond).count();


            double max_diag = 0, min_diag = 1e100;
            for (int i = 0; i < n_random; i++) {
                double d = mme.ZtZ_Ginv.coeff(i, i);
                if (d > max_diag) max_diag = d;
                if (d < min_diag && d > 0) min_diag = d;
            }
            if (config.verbose) {
                std::cout << "ZtZ_Ginv diagonal range: [" << min_diag << ", " << max_diag << "]\n";
                std::cout << "Ratio: " << (max_diag / min_diag) << "\n";
            }

            
            // Phase 3: Hutchinson++ diagonal estimation
            if (config.verbose) std::cout << "\n[3] Computing diagonal with Hutchinson++ Block-PCG...\n";
            auto t_solve = std::chrono::high_resolution_clock::now();
            
            
            int actual_samples = 0;
            int num_vectors_per_batch = 4;
            HutchinsonPlusPlus hutch_pp(mme_size, num_vectors_per_batch, config.threads,
                            config.block_size, n_fixed, n_random);
            
            int batch = 0;
            double max_err = 1.0;
            
            while (actual_samples < config.max_samples && max_err > config.convergence_tol) {
                if (config.verbose && batch % 5 == 0) {
                    std::cout << "  Batch " << batch << ": generating vectors...\n" << std::flush;
                }
                hutch_pp.generateRandomVectors(batch, perf_mon);
                
                if (config.verbose && batch % 5 == 0) {
                    std::cout << "  Batch " << batch << ": solving PCG...\n" << std::flush;
                }
                auto solve_start = std::chrono::high_resolution_clock::now();
                hutch_pp.solveBatch(block_pcg, config.pcg_tol, config.pcg_maxiter, perf_mon);
                auto solve_end = std::chrono::high_resolution_clock::now();
    
                if (config.verbose && batch % 5 == 0) {
                    double solve_time = std::chrono::duration<double>(solve_end - solve_start).count();
                    std::cout << "  Batch " << batch << ": PCG took " << solve_time 
                              << "s, updating estimates...\n" << std::flush;
                }
                hutch_pp.updateEstimates(perf_mon);
    
                actual_samples += num_vectors_per_batch;
                batch++;
                

            if (actual_samples >= 30 && actual_samples % 10 == 0) {
                max_err = hutch_pp.getMaxError(actual_samples);
    
                if (config.verbose && actual_samples % 20 == 0) {
                    std::cout << "  Sample " << actual_samples 
                              << ": max error = " << std::scientific << max_err << "\n";
                }
    
                if (max_err < config.convergence_tol) {
                    if (config.verbose) {
                        std::cout << "  Converged at sample " << actual_samples << "\n";
                    }
                    break;
                }
            }}


                        
            timings["stochastic_diag"] = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_solve).count();
            
            // Phase 4: Post-processing
            if (config.verbose) std::cout << "\n[4] Post-processing results...\n";
            auto t_post = std::chrono::high_resolution_clock::now();


            if (config.block_size > 1) {
                hutch_pp.symmetrize(actual_samples);
            }


            hutch_pp.writeResults(config.output_file, actual_samples);


                
            timings["post_process"] = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t_post).count();
            
            auto total_end = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double>(total_end - total_start).count();
            
            if (config.verbose) {
                perf_mon.report();
                
                std::cout << "\n========================================\n";
                std::cout << "  Computation Complete\n";
                std::cout << "========================================\n";
                std::cout << "Timings (seconds):\n";
                for (const auto& t : timings) {
                    std::cout << "  " << t.first << ": " << std::fixed << std::setprecision(3) << t.second << "\n";
                }
                std::cout << "  Total time: " << total_time << "\n";
                std::cout << "\nOutput file: " << config.output_file << "\n";
                std::cout << "========================================\n";
            }
if (config.compute_exact) {
    try {
        // 确定要计算的动物ID
        std::vector<int> animal_ids;
        int n_animals = n_random / config.block_size;
        int n_compute = std::min(config.exact_n_animals, n_animals);
        
        for (int i = 0; i < n_compute; i++) {
            animal_ids.push_back(i);
        }
  
        auto params = FastMMEBuilder::readParametersFast();
        auto precomp = FastMMEBuilder::precomputeDataStructures(params);
     
        if (config.verbose) {
            std::cout << "\n[Exact Computation Requested]\n";
            std::cout << "Computing exact inverse for " << n_compute 
                      << " animals...\n";
        }
        
        // 计算精确块
        auto exact_blocks = ExactBlockInverter::computeExactBlocks(
            A, animal_ids, config.block_size, n_fixed, 
            precomp, params,  // 传入这两个参数
            config.verbose
        );        
        // 输出到文件
        ExactBlockInverter::writeResults(
            exact_blocks, animal_ids, n_fixed, config.block_size, 
            config.exact_output
        );
        
        if (config.verbose) {
            std::cout << "\n[Exact Computation Complete]\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nWARNING: Exact computation failed: " << e.what() << "\n";
        std::cerr << "Continuing without exact results.\n";
    }
}

            
            return true;
            
        } catch(const std::exception& e) {
            std::cerr << "\nERROR: " << e.what() << "\n";
            return false;
        } catch(...) {
            std::cerr << "\nERROR: Unknown exception occurred\n";
            return false;
        }
    }
    
};

// =============================================================================
// Main
// =============================================================================
int main(int argc, char* argv[]) {
    MMEDiagonalSolver::Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--max-samples" && i + 1 < argc) {
            config.max_samples = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.threads = std::atoi(argv[++i]);
        } else if (arg == "--conv-tol" && i + 1 < argc) {
            config.convergence_tol = std::atof(argv[++i]);
        } else if (arg == "--regularization" && i + 1 < argc) {
            config.regularization = std::atof(argv[++i]);        
        } else if (arg == "--pcg-tol" && i + 1 < argc) {
            config.pcg_tol = std::atof(argv[++i]);
        } else if (arg == "--pcg-maxiter" && i + 1 < argc) {
            config.pcg_maxiter = std::atoi(argv[++i]);
        } else if (arg == "--block-size" && i + 1 < argc) {
                  config.block_size = std::atoi(argv[++i]);
        } else if (arg == "--exact") {
            config.compute_exact = true;
        } else if (arg == "--exact-animals" && i + 1 < argc) {
            config.exact_n_animals = std::atoi(argv[++i]);
        } else if (arg == "--exact-output" && i + 1 < argc) {
    config.exact_output = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_file = argv[++i];
        } else if (arg == "--quiet") {
            config.verbose = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "\nRequired files:\n";
            std::cout << "  blockM.par         : Parameter file\n";
            std::cout << "  null.add.sum       : Miss address sum file\n";
            std::cout << "  (data file specified in blockM.par)\n";
            std::cout << "\nOptional arguments:\n";
            std::cout << "  --max-samples N       : Maximum number of samples (default: 200)\n";
            std::cout << "  --threads N           : Number of threads (default: auto)\n";
            std::cout << "  --conv-tol TOL        : Convergence tolerance (default: 0.001)\n";
            std::cout << "  --pcg-tol TOL         : PCG tolerance (default: 1e-6)\n";
            std::cout << "  --pcg-maxiter N       : PCG max iterations (default: 300)\n";
            std::cout << "  --output FILE         : Output file name (default: mme_diagonal_results.txt)\n";
            std::cout << "  --block-size N        : Block size for block diagonal estimation (default: 1)\n";
            std::cout << "                          1 = scalar mode, >1 = block mode (e.g., 5 for 5x5 blocks)\n";
            std::cout << "  --quiet               : Suppress verbose output\n";
            return 0;
        }
    }
    
    MMEDiagonalSolver solver;
    solver.setConfig(config);
    
    bool success = solver.solve();
    
    return success ? 0 : 1;
}