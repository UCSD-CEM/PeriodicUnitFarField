#pragma once

#include "cusp/coo_matrix.h"
#include "cusp/multiply.h"
#include "cusp/transpose.h"
#include "cusp/sort.h"
#include "cusp/print.h"
#include <thrust/device_ptr.h>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <limits>

#define INDEX_TYPE size_t // We pre-define the IndexType as a size_t for maximal compatibility
static std::string delimiter = "*";

namespace puff {

    template<typename IndexType, typename ValueType, typename MemorySpace>
    using SparseMatrix = cusp::coo_matrix<IndexType, ValueType, MemorySpace>;        

    template<typename ValueType, typename MemorySpace>
    using Vector = cusp::array1d<ValueType, MemorySpace>;

    template<typename IndexType, typename ValueType, typename MemorySpace>
    struct SparseMatrixViewHelper;

    template<typename IndexType, typename ValueType>
    struct SparseMatrixViewHelper<IndexType, ValueType, cusp::host_memory> {
        using type = decltype(cusp::make_coo_matrix_view(
            std::declval<size_t>(), std::declval<size_t>(), std::declval<size_t>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<IndexType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<IndexType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<ValueType *>, 
            thrust::detail::normal_iterator<IndexType *>>>>()
        ));
    };

    template<typename IndexType, typename ValueType>
    struct SparseMatrixViewHelper<IndexType, ValueType, cusp::device_memory> {
        using type = decltype(cusp::make_coo_matrix_view(
            std::declval<size_t>(), std::declval<size_t>(), std::declval<size_t>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>(),
            std::declval<cusp::array1d_view<thrust::permutation_iterator<
            thrust::detail::normal_iterator<thrust::device_ptr<ValueType>>, 
            thrust::detail::normal_iterator<thrust::device_ptr<IndexType>>>>>()
        ));
    };

    template<typename IndexType, typename ValueType, typename MemorySpace>
    using SparseMatrixView = typename SparseMatrixViewHelper<IndexType, ValueType, MemorySpace>::type;


    template<typename IndexType, typename ValueType, typename MemorySpace>
    class SparseMatrixWrapper {
        public:
            SparseMatrixWrapper() {}

            SparseMatrixWrapper(SparseMatrix<IndexType, ValueType, MemorySpace> matrix) {
                matrix = matrix;
            }
            
            size_t get_num_rows() {
                return matrix.num_rows;
            }

            size_t get_num_cols() {
                return matrix.num_cols;
            }

            size_t get_num_entries() {
                return matrix.num_entries;
            }


            void insert_entry(IndexType row, IndexType col, ValueType val) {
                std::string row_col_string = row_col_to_string(row, col);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    entries[row_col_string] = val;
                }
            }

            void remove_entry(IndexType row, IndexType col) {
                std::string row_col_string = row_col_to_string(row, col);
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    entries.erase(row_col_string);
                }
            }

            void make_matrix()
            {
                std::lock_guard<std::mutex> lock(mtx);
                Vector<IndexType, cusp::host_memory> h_I;
                Vector<IndexType, cusp::host_memory> h_J;
                Vector<ValueType, cusp::host_memory> h_V;
                for(auto& [row_col_string, value] : entries)
                {
                    auto [row, col] = string_to_row_col(row_col_string);
                    if(value == ValueType(0)) continue; // No 0 element
                    h_I.push_back(row);
                    h_J.push_back(col);
                    h_V.push_back(value);
                }
                
                
                // sort triplets by (i,j) index using two stable sorts (first by J, then by I)
                thrust::stable_sort_by_key(h_J.begin(), h_J.end(), thrust::make_zip_iterator(thrust::make_tuple(h_I.begin(), h_V.begin())));
                thrust::stable_sort_by_key(h_I.begin(), h_I.end(), thrust::make_zip_iterator(thrust::make_tuple(h_J.begin(), h_V.begin())));


                // Calculate num_Rows
                IndexType numRows = *thrust::max_element(h_I.begin(), h_I.end()) + 1; // Assume row indices are 0-based
                IndexType numCols = *thrust::max_element(h_J.begin(), h_J.end()) + 1; // Assume column indices are 0-based
                /*****************CSR Matrix********************/
                /*
                Vector<IndexType, cusp::host_memory> h_row_ptrs(numRows + 1, 0); // Initialize row pointers with zeros
                IndexType currentIndex = 0;
                IndexType currentRow = std::numeric_limits<IndexType>::max();
                for (IndexType i = 0; i < h_I.size(); ++i) {
                    if (h_I[i] != currentRow) { // New row encountered
                        currentRow = h_I[i];
                        h_row_ptrs[currentRow] = currentIndex; // Update row pointer for the new row
                    }
                    currentIndex++;
                }
                h_row_ptrs[numRows] = h_I.size(); // Set the last element of row_ptrs

                // Move to prescribed memory space
                Vector<IndexType, MemorySpace> I(h_row_ptrs);
                Vector<IndexType, MemorySpace> J(h_J);
                Vector<ValueType, MemorySpace> V(h_V);

                // resize matrix
                matrix.resize(numRows, numCols, V.size());

                // Insert I to matrix.row_offsets
                thrust::copy(I.begin(), I.end(), matrix.row_offsets.begin());
                // Insert J to matrix.column_indices
                thrust::copy(J.begin(), J.end(), matrix.column_indices.begin());
                // Insert V to matrix.values
                thrust::copy(V.begin(), V.end(), matrix.values.begin());
                */

                /*****************COO Matrix********************/
                // Move to prescribed memory space
                Vector<IndexType, MemorySpace> I(h_I);
                Vector<IndexType, MemorySpace> J(h_J);
                Vector<ValueType, MemorySpace> V(h_V);

                // resize matrix
                matrix.resize(numRows, numCols, V.size());

                // Insert I to matrix.row_offsets
                thrust::copy(I.begin(), I.end(), matrix.row_indices.begin());
                // Insert J to matrix.column_indices
                thrust::copy(J.begin(), J.end(), matrix.column_indices.begin());
                // Insert V to matrix.values
                thrust::copy(V.begin(), V.end(), matrix.values.begin());

                // Make the transpose coo_matrix view
                permutation = Vector<IndexType, MemorySpace>(cusp::counting_array<IndexType>(matrix.num_entries));
                Vector<IndexType, MemorySpace> matrix_column_indices(matrix.column_indices);
                cusp::counting_sort_by_key(matrix_column_indices, permutation, IndexType(0), IndexType(matrix.num_cols));
                matrix_t = cusp::make_coo_matrix_view(matrix.num_rows, matrix.num_cols, matrix.num_entries,
                                                cusp::make_array1d_view(thrust::make_permutation_iterator(matrix.column_indices.begin(), permutation.begin()),
                                                                    thrust::make_permutation_iterator(matrix.column_indices.begin(), permutation.end())),
                                                cusp::make_array1d_view(thrust::make_permutation_iterator(matrix.row_indices.begin(),    permutation.begin()),
                                                                    thrust::make_permutation_iterator(matrix.row_indices.begin(),    permutation.end())),
                                                cusp::make_array1d_view(thrust::make_permutation_iterator(matrix.values.begin(),         permutation.begin()),
                                                                      thrust::make_permutation_iterator(matrix.values.begin(),         permutation.end())));
            }

            void reset()
            {
                std::lock_guard<std::mutex> lock(mtx);
                matrix.resize(0, 0, 0);
                entries.clear();
            }

            void print_matrix() {
                std::lock_guard<std::mutex> lock(mtx);
                cusp::print(matrix);
            }

            void SpMV(Vector<ValueType, MemorySpace>& x, Vector<ValueType, MemorySpace>& y, bool transpose = false) {            
                if(&x == &y) {
                    Vector<ValueType, MemorySpace> temp(x.size());
                    if (transpose) {
                        cusp::multiply(matrix_t, x, temp);
                    } else {
                        cusp::multiply(matrix, x, temp);
                    }
                    y.swap(temp);
                    return;
                }

                if (transpose) {
                    cusp::multiply(matrix_t, x, y);
                } else {
                    cusp::multiply(matrix, x, y);
                }
            }


            void SpMVP(ValueType alpha, Vector<ValueType, MemorySpace>& x, ValueType beta, Vector<ValueType, MemorySpace>& y, bool transpose = false) {
                // y = alpha * A * x + beta * y
                if (beta == 0)
                {
                    // y = A * x
                    if(alpha != 0) SpMV(x, y, transpose); 
                    // y *= alpha;
                    if(alpha != 1)
                        thrust::transform(y.begin(), y.end(), thrust::make_constant_iterator(alpha), y.begin(), thrust::multiplies<ValueType>());
                }
                else
                {
                    Vector<ValueType, MemorySpace> temp(x.size(), 0);
                    // temp = A * x
                    if(alpha != 0) SpMV(x, temp, transpose); 
                    // temp = alpha * temp
                    if(alpha != 0 && alpha != 1)
                        thrust::transform(temp.begin(), temp.end(), thrust::make_constant_iterator(alpha), temp.begin(), thrust::multiplies<ValueType>());
                    // y = beta * y
                    if(beta != 1)
                        thrust::transform(y.begin(), y.end(), thrust::make_constant_iterator(beta), y.begin(), thrust::multiplies<ValueType>());
                    // y = temp + y
                    thrust::transform(temp.begin(), temp.end(), y.begin(), y.begin(), thrust::plus<ValueType>());
                }
            }

            
        private:
            SparseMatrix<IndexType, ValueType, MemorySpace> matrix;
            // Transpose matrix view
            // Use decltype to infer the type
            SparseMatrixView<IndexType, ValueType, MemorySpace> matrix_t;
            Vector<IndexType, MemorySpace> permutation; // permutation for transpose

            std::unordered_map<std::string, ValueType> entries; // Use string for better hashing without potential hash collision
            // mutex lock
            std::mutex mtx;

            std::string row_col_to_string(IndexType row, IndexType col) {
                return std::to_string(row) + delimiter + std::to_string(col); //seperate by space
            }

            std::pair<IndexType, IndexType> string_to_row_col(const std::string& row_col_string)
            {
                std::string row_string = row_col_string.substr(0, row_col_string.find(delimiter));
                std::string col_string = row_col_string.substr(row_col_string.find(delimiter) + 1, row_col_string.length());
                IndexType row = 0;
                IndexType col = 0;
                for(IndexType i = 0; i < row_string.length(); i++) {
                    row = row * 10 + (row_string[i] - '0');
                }
                for(IndexType i = 0; i < col_string.length(); i++) {
                    col = col * 10 + (col_string[i] - '0');
                }
                return std::make_pair(row, col);
            }
            
    };






    template<typename ValueType>
    using SparseMatrix_h = SparseMatrixWrapper<INDEX_TYPE, ValueType, cusp::host_memory>;

    template<typename ValueType>
    using SparseMatrix_d = SparseMatrixWrapper<INDEX_TYPE, ValueType, cusp::device_memory>;

    template<typename ValueType>
    using Vector_h = Vector<ValueType, cusp::host_memory>;

    template<typename ValueType>
    using Vector_d = Vector<ValueType, cusp::device_memory>;
    
}
