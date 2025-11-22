//
// Created by asus on 2025/4/12.
//

#ifndef MERGE_SORT_CUH
#define MERGE_SORT_CUH
#include <cuda_runtime.h>

namespace cuda_sort
{
    template <typename T, typename Comp>
    __device__ void merge(T* arr, T* temp, std::size_t left, std::size_t mid, std::size_t right, Comp comp)
    {
        std::size_t i = left;
        std::size_t j = mid;
        std::size_t k = left;

        while (i < mid && j < right)
        {
            if (comp(arr[i], arr[j]))
            {
                temp[k++] = arr[i++];
            }
            else
            {
                temp[k++] = arr[j++];
            }
        }

        while (i < mid)
            temp[k++] = arr[i++];
        while (j < right)
            temp[k++] = arr[j++];
    }

    template <typename T, typename Comp>
    __global__ void merge_kernel(T* arr, T* temp, std::size_t n, int width, Comp comp)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t left = 2 * idx * width;
        std::size_t mid = std::min(left + width, n);
        std::size_t right = std::min(left + 2 * width, n);

        if (left < n)
        {
            merge(arr, temp, left, mid, right, comp);
        }
    }

    namespace details
    {
        constexpr unsigned MergeSharedSizeLimit = 1024;
        constexpr unsigned MergeSampleStride = 128;
        constexpr unsigned MergeSampleThreads = 256;

        /**
         * @brief calculate a / b for integer, if a % b == 0, then return a/b, else return a/b+1
         * @tparam T integer type
         * @param a dividend
         * @param b divisor
         * @return a/b if a%b == 0, else a/b + 1
         */
        template <typename T>
        __host__ __device__ T i_div_up(const T a, const T b)
        {
            return ((a % b) == 0) ? (a / b) : (a / b + 1);
        }

        /// return dividend / MergeSampleStride
        template <typename T>
        __host__ __device__ T get_sample_count(T dividend)
        {
            return i_div_up(dividend, MergeSampleStride);
        }

        /// return the next minimal power of 2, for example, `next_power_of2(9) = 16`
        template <typename T>
        __host__ __device__ T next_power_of2(T x)
        {
            return static_cast<T>(1) << ((sizeof(T) * 8) - __clz(x - 1));
        }

        /**
         * @brief Perform a binary search to find the inclusive upper bound for a value in a sorted array.
         *
         * This device function finds the largest index `pos` such that all elements in `data[0...pos-1]`
         * are <= `val`, using a stride-based binary search algorithm. Supports custom comparators for
         * flexible ordering.
         *
         * @tparam T     Element type (must support comparison via `Comp`)
         * @tparam Comp  Comparison functor type (default: std::less<>)
         *
         * @param val    Target value to search for
         * @param data   Sorted input array (non-decreasing order)
         * @param len    Number of elements in the array
         * @param stride Initial search stride (typically power-of-2 <= len)
         * @param comp   Comparison functor instance (default-constructed)
         *
         * @return The insertion index `pos` where:
         *         - `0 <= pos <= len`
         *         - All elements before `pos` are <= `val`
         *         - Returns 0 if `len == 0`
         *
         * @details
         * Algorithm steps:
         * 1. Start with initial stride (e.g., len/2)
         * 2. Iteratively halve the stride while checking elements
         * 3. Expand search window when elements <= val are found
         * 4. Guarantees O(log n) time complexity
         *
         * Key properties:
         * - Exponential search variant optimized for GPU execution
         * - Maintains warp convergence through synchronous memory access patterns
         * - Supports non-power-of-2 array lengths
         *
         * @note
         * - Requires `data` to be pre-sorted according to `comp`
         * - Typically used in parallel sorting/partitioning algorithms
         * - `stride` should be initialized to the largest power-of-2 less than `len`
         *
         * @example
         * ```
         * __device__ void example() {
         *     int data[] = {1, 3, 5, 7, 9};
         *     unsigned pos = binary_search_inclusive(6, data, 5, 4); // returns 3
         * }
         * ```
         */
        template <typename T, typename Comp = std::less<>>
        __device__ unsigned binary_search_inclusive(T val, const T* data, const unsigned len, unsigned stride,
                                                    Comp comp = {})
        {
            if (len == 0) return {};
            unsigned pos = 0;
            for (; stride > 0; stride >>= 1)
            {
                auto newPos = std::min(pos + stride, len);
                if (comp(data[newPos - 1], val) || data[newPos - 1] == val)
                {
                    pos = newPos;
                }
            }
            return pos;
        }

        /**
         * @brief CUDA kernel to perform merge sort using shared memory optimization.
         *
         * This kernel implements a parallel merge sort algorithm using a bottom-up approach.
         * Each thread block processes a segment of data stored in shared memory, performing
         * iterative pairwise merges with binary search for element placement. The algorithm
         * minimizes global memory access by leveraging shared memory for intermediate results.
         *
         * @tparam T     Data type to be sorted (must support comparison via `Comp`)
         * @tparam Comp  Comparison functor type (default: std::less<>)
         *
         * @param src    [in] Input data array (global memory)
         * @param dst    [out] Output sorted array (global memory)
         * @param size   Total number of elements to sort
         * @param comp   Comparison functor instance (default-constructed)
         *
         * @details
         * Algorithm workflow:
         * 1. Load data into shared memory (2x block size for efficient merging)
         * 2. Iteratively merge subarrays with exponentially increasing strides:
         *    - Use binary search to find insertion positions
         *    - Perform parallel merge using warp-synchronous operations
         * 3. Write final sorted results back to global memory
         *
         * Key optimizations:
         * - Shared memory reduces global memory bandwidth usage [[2]][[8]]
         * - Binary search (`binary_search_inclusive`) enables O(log n) merge steps [[4]][[7]]
         * - Warp-level synchronization (`__syncthreads()`) ensures data consistency [[3]][[9]]
         *
         * @note
         * - Requires `size` to be power-of-2 for full efficiency
         * - Shared memory size must be at least `2 * MergeSharedSizeLimit`
         * - Should be launched with `MergeSharedSizeLimit/2` threads per block
         */
        template <typename T, typename Comp = std::less<>>
        __global__ void merge_sort_shared_kernel(T* src, T* dst, unsigned size, Comp comp = {})
        {
            __shared__ T sharedKey[MergeSharedSizeLimit];
            auto idx = blockIdx.x * MergeSharedSizeLimit + threadIdx.x;
            auto tid = threadIdx.x;
            // Load data into shared memory (2x block size for merging)
            sharedKey[tid] = src[idx];
            sharedKey[tid + MergeSharedSizeLimit / 2] = src[idx + MergeSharedSizeLimit / 2];
            // Merge phases with exponentially increasing stride
            for (unsigned stride = 1; stride < size; stride *= 2)
            {
                auto pos = tid & (stride - 1); // tid % stride
                T* baseKey = sharedKey + 2 * (tid - pos);

                __syncthreads();
                // Perform binary search to find merge positions
                auto keyA = baseKey[pos];
                auto keyB = baseKey[pos + stride];
                auto posA = binary_search_inclusive(keyA, baseKey + stride, stride, stride, comp) + pos;
                auto posB = binary_search_inclusive(keyB, baseKey + 0, stride, stride, comp) + pos;
                __syncthreads();
                baseKey[posA] = std::move(keyA);
                baseKey[posB] = std::move(keyB);
            }
            __syncthreads();
            dst[idx] = std::move(sharedKey[tid]);
            dst[idx + MergeSharedSizeLimit / 2] = std::move(sharedKey[tid + MergeSharedSizeLimit / 2]);
        }

        /**
         * @brief Host-side wrapper to launch shared-memory merge sort kernel.
         *
         * Configures and launches the CUDA kernel for merge sort using shared memory optimization.
         * Automatically calculates grid and block dimensions based on input parameters.
         *
         * @tparam T     Data type to be sorted
         * @tparam Comp  Comparison functor type
         *
         * @param src        [in] Input data array (global memory)
         * @param dst        [out] Output sorted array (global memory)
         * @param batchSize  Number of independent sort batches (each of length `len`)
         * @param len        Length of each batch to sort (must be power-of-2)
         * @param comp       Comparison functor instance (default-constructed)
         *
         * @details
         * Configuration rules:
         * - Total elements per kernel launch: `batchSize * len`
         * - Block size: `MergeSharedSizeLimit / 2` threads
         * - Grid size: `(batchSize * len) / MergeSharedSizeLimit` blocks
         *
         * @note
         * - Returns immediately if `len < 2` (no sorting needed)
         * - Requires `MergeSharedSizeLimit` to be a power-of-2
         * - Performance scales with shared memory bandwidth [[2]][[8]]
         */
        template <typename T, typename Comp = std::less<>>
        void merge_sort_shared(T* src, T* dst, const unsigned batchSize, unsigned len, Comp comp = {})
        {
            if (len < 2) return;
            auto blockCount = batchSize * len / MergeSharedSizeLimit;
            auto threadCount = MergeSharedSizeLimit / 2;
            merge_sort_shared_kernel<<<blockCount, threadCount>>>(src, dst, len, comp);
        }

        /**
         * @brief CUDA kernel to generate sample ranks for merge-based sorting algorithms.
         *
         * This kernel generates rank tables for two input sequences (A and B) by performing
         * binary searches to determine cross-sequence element positions. The results are used
         * to optimize large-scale merge operations in GPU sorting algorithms.
         *
         * @tparam T     Element type (must support comparison via `Comp`)
         * @tparam Comp  Comparison functor type (default: std::less<>)
         *
         * @param ranksA    [out] Rank table for sequence A (element positions in B)
         * @param ranksB    [out] Rank table for sequence B (element positions in A)
         * @param src       [in] Source data array containing both sequences
         * @param stride    Base offset between sequence A and B in `src`
         * @param len       Total length of both sequences
         * @param threadCount Number of active threads required for processing
         * @param comp      Comparison functor instance (default-constructed)
         *
         * @details
         * Processing steps per thread:
         * 1. Calculate segment base index based on thread position
         * 2. Determine segment boundaries for sequences A and B
         * 3. For elements in A:
         *    - Record their original index in `ranksA`
         *    - Use binary search to find their position in sequence B
         * 4. For elements in B:
         *    - Record their original index in `ranksB`
         *    - Use binary search to find their position in sequence A
         * 5. Key optimizations:
         *    - Coalesced memory access patterns
         *    - Warp-specialized binary search via `binary_search_inclusive`
         *    - Double-buffering through separate rank tables
         *
         * @note
         * - `stride` must be power-of-2 for correct segment alignment
         * - Requires `len` to be multiple of `MergeSampleStride` for full efficiency
         */
        template <typename T, typename Comp = std::less<>>
        __global__ void generate_sample_ranks_kernel(unsigned* ranksA, unsigned* ranksB, T* src, unsigned stride,
                                                     const unsigned len, const unsigned threadCount, Comp comp = {})
        {
            const auto pos = blockIdx.x * blockDim.x + threadIdx.x;
            if (pos >= threadCount) return;
            const auto idx = pos & ((stride / MergeSampleStride) - 1);
            const auto segmentBase = (pos - idx) * (2 * MergeSampleStride);

            const unsigned segmentElementsA = stride;
            const unsigned segmentElementsB = std::min(stride, len - segmentBase - stride);
            const unsigned segmentSampleA = get_sample_count(segmentElementsA);
            const unsigned segmentSampleB = get_sample_count(segmentElementsB);

            if (idx < segmentElementsA)
            {
                ranksA[idx + segmentBase / MergeSampleStride] = idx * MergeSampleStride;
                ranksB[idx + segmentBase / MergeSampleStride] = binary_search_inclusive(
                    src[idx * MergeSampleStride + segmentBase], src + stride + segmentBase,
                    segmentElementsB, next_power_of2(segmentElementsB), comp
                );
            }

            if (idx < segmentElementsB)
            {
                ranksB[idx + segmentBase / MergeSampleStride + stride / MergeSampleStride] = idx * MergeSampleStride;
                ranksA[idx + segmentBase / MergeSampleStride + stride / MergeSampleStride] = binary_search_inclusive(
                    src[idx * MergeSampleStride + segmentBase + stride], src + segmentBase,
                    segmentElementsA, next_power_of2(segmentElementsA), comp
                );
            }
        }

        /**
         * @brief Host-side wrapper to launch sample rank generation kernel.
         *
         * Configures and launches the CUDA kernel for generating merge sample ranks.
         * Automatically calculates grid dimensions based on input parameters and
         * hardware constraints.
         *
         * @tparam T     Element type
         * @tparam Comp  Comparison functor type
         *
         * @param ranksA    [out] Rank table for sequence A
         * @param ranksB    [out] Rank table for sequence B
         * @param src       [in] Source data array containing both sequences
         * @param stride    Base offset between sequences in `src`
         * @param len       Total length of both sequences
         * @param comp      Comparison functor instance (default-constructed)
         *
         * @details
         * Configuration logic:
         * 1. Calculate processing threads based on segment sizes
         * 2. Handle edge cases where last segment has uneven elements
         * 3. Launch kernel with optimal block size (`MergeSampleThreads`)
         *
         * @note
         * - Automatically handles partial segments at the end of the array
         * - Requires `stride` to be power-of-2 for correct operation
         * - Performance scales with `MergeSampleThreads` configuration [[2]][[6]]
         */
        template <typename T, typename Comp = std::less<>>
        void generate_sample_ranks(unsigned* ranksA, unsigned* ranksB, T* src, unsigned stride, unsigned len,
                                   Comp comp = {})
        {
            const auto lastSegmentElements = len % (2 * stride);
            auto threadCount = (lastSegmentElements > stride)
                                   ? (len + 2 * stride - lastSegmentElements) / (2 * MergeSampleStride)
                                   : (len - lastSegmentElements) / (2 * MergeSampleStride);
            generate_sample_ranks_kernel<<<i_div_up(threadCount, MergeSampleThreads), MergeSampleThreads>>>(
                ranksA, ranksB, src, stride, len, threadCount, comp);
        }

        /**
         * @brief CUDA kernel to merge rank information with global indices for merge-based sorting.
         *
         * This kernel processes precomputed rank tables to generate limit tables that define
         * the boundaries between merged segments. It uses binary search to efficiently determine
         * cross-segment element positions in O(log n) time per element.
         *
         * @param limits      [out] Output limit table storing merged boundary indices
         * @param ranks       [in] Input rank table containing precomputed element positions
         * @param stride      Base offset between segments in the rank table
         * @param len         Total length of the original data array
         * @param threadCount Number of active threads required for processing
         *
         * @details
         * Algorithm workflow:
         * 1. Calculate segment base index for current thread
         * 2. Process elements in both segments (A and B) of the current partition:
         *    - For each element in segment A:
         *      - Use binary search to find its position in segment B's rank table
         *      - Store merged position in limit table
         *    - For each element in segment B:
         *      - Use binary search to find its position in segment A's rank table
         *      - Store merged position in limit table
         * 3. Results are used to construct non-overlapping intervals for final merge
         *
         * Key optimizations:
         * - Coalesced memory access through strided indexing [[2]][[8]]
         * - Warp-specialized binary search via `binary_search_inclusive` [[4]]
         * - Double-buffering through separate limit tables [[5]]
         *
         * @note
         * - `stride` must be power-of-2 for correct segment alignment
         * - Requires precomputed rank tables from `generate_sample_ranks`
         */
        __global__ void merge_ranks_and_indices_kernel(unsigned* limits, const unsigned* ranks,
                                                       unsigned stride, const unsigned len,
                                                       const unsigned threadCount)
        {
            const auto pos = blockIdx.x * blockDim.x + threadIdx.x;
            if (pos >= threadCount) return;
            // Calculate segment base and indexes (segmentBase is global data offset)
            const auto i = pos & ((stride / MergeSampleStride) - 1); // Current element index within segment
            const auto segmentBase = (pos - i) * (2 * MergeSampleStride); // Base offset for current segment pair
            const auto rankBase = (pos - i) * 2; // Base index in rank table
            const auto limitBase = (pos - i) * 2; // Base index in limit table

            // Determine segment sizes for A and B sequences
            const auto segmentElementA = stride;
            const auto segmentElementB = std::min(stride, len * segmentBase - stride);
            const auto segmentSamplesA = get_sample_count(segmentElementA);
            const auto segmentSamplesB = get_sample_count(segmentElementB);

            // Process elements in segment A
            if (i < segmentElementA)
            {
                // Find insertion position in B's rank table using binary search
                auto dstPos = binary_search_inclusive(ranks[rankBase + i], ranks + rankBase + segmentSamplesA,
                                                      segmentSamplesB, next_power_of2(segmentSamplesB),
                                                      std::less<>{}) + i;
                // Store merged position
                limits[limitBase + dstPos] = ranks[rankBase + i];
            }
            // Process elements in segment B
            if (i < segmentElementB)
            {
                auto dstPos = binary_search_inclusive(
                    ranks[rankBase + segmentElementA + i], ranks + rankBase, segmentElementA,
                    next_power_of2(segmentElementA)) + i;
                limits[limitBase + dstPos] = ranks[rankBase + segmentElementA + i];
            }
        }

        /**
         * @brief Host-side wrapper to launch rank merging kernel for two rank tables.
         *
         * Configures and launches the CUDA kernel to merge rank information from two
         * precomputed rank tables (A and B) into corresponding limit tables.
         *
         * @param limitsA  [out] Limit table for sequence A
         * @param limitsB  [out] Limit table for sequence B
         * @param ranksA   [in] Rank table for sequence A
         * @param ranksB   [in] Rank table for sequence B
         * @param stride   Base offset between segments in rank tables
         * @param len      Total length of the original data array
         *
         * @details
         * Processing strategy:
         * 1. Handle partial segments at the end of the array
         * 2. Launch separate kernels for each rank table (A and B)
         * 3. Use optimal thread configuration (`MergeSampleThreads`) for performance
         *
         * @note
         * - Automatically handles edge cases with uneven segment lengths
         * - Requires `stride` to be power-of-2 for correct operation
         * - Performance scales with `MergeSampleThreads` configuration [[2]][[6]]
         */
        void merge_ranks_and_indices(unsigned* limitsA, unsigned* limitsB, unsigned* ranksA,
                                     unsigned* ranksB, unsigned stride, unsigned len)
        {
            auto lastSegmentElements = len % (2 * stride);
            auto threadCount = (lastSegmentElements > stride)
                                   ? (len + 2 * stride - lastSegmentElements) / (2 * MergeSampleStride)
                                   : (len - lastSegmentElements) / (2 * MergeSampleStride);
            merge_ranks_and_indices_kernel<<<i_div_up(threadCount, MergeSampleThreads), MergeSampleThreads>>>(
                limitsA, ranksA, stride, len, threadCount);
            merge_ranks_and_indices_kernel<<<i_div_up(threadCount, MergeSampleThreads), MergeSampleThreads>>>(
                limitsB, ranksB, stride, len, threadCount);
        }

        /**
         * @brief Device-side merge implementation for two sorted sequences.
         *
         * Merges two sorted subarrays (A and B) into a single sorted array using binary search
         * to determine element positions. This function is optimized for warp-level parallelism
         * and shared memory efficiency.
         *
         * @tparam T     Element type (must support comparison via `Comp`)
         * @tparam Comp  Comparison functor type (default: std::less<>)
         *
         * @param srcA          [in] First sorted subarray (A)
         * @param srcB          [in] Second sorted subarray (B)
         * @param dst           [out] Merged output array
         * @param lenA          Length of subarray A
         * @param nPowTwoLenA   Next power-of-2 length for subarray A (for alignment)
         * @param lenB          Length of subarray B
         * @param nPowTwoLenB   Next power-of-2 length for subarray B (for alignment)
         * @param comp          Comparison functor instance (default-constructed)
         *
         * @details
         * Algorithm workflow:
         * 1. Each thread loads one element from A or B (if in bounds)
         * 2. Use binary search to find insertion positions in the opposite subarray
         * 3. Write elements to the final merged position in shared memory
         *
         * Key optimizations:
         * - Warp-synchronous execution minimizes thread divergence [[3]][[9]]
         * - Shared memory reduces global memory access latency [[2]][[8]]
         * - Coalesced memory access through strided indexing [[6]]
         *
         * @note
         * - Requires `nPowTwoLenA` and `nPowTwoLenB` to be power-of-2 for binary search alignment
         * - Threads out of bounds write dummy values (safe due to guard conditions)
         */
        template <typename T, typename Comp = std::less<>>
        __device__ void merge(T* srcA, T* srcB, T* dst, unsigned lenA, unsigned nPowTwoLenA, unsigned lenB,
                              unsigned nPowTwoLenB, Comp comp = {})
        {
            auto tid = threadIdx.x;
            auto keyA = tid < lenA ? srcA[tid] : T{0};
            auto keyB = tid < lenB ? srcB[tid] : T{0};
            auto dstPosA = tid < lenA ? binary_search_inclusive(keyA, srcB, lenB, nPowTwoLenB, comp) : T{0};
            auto dstPosB = tid < lenB ? binary_search_inclusive(keyB, srcA, lenA, nPowTwoLenA, comp) : T{0};

            __syncthreads();
            if (tid < lenA)
            {
                dst[dstPosA] = std::move(keyA);
            }

            if (tid < lenB)
            {
                dst[dstPosB] = std::move(keyB);
            }
        }

        /**
         * @brief CUDA kernel to merge elementary intervals using precomputed rank tables.
         *
         * This kernel merges adjacent sorted segments by processing intervals defined by
         * limit tables (`limitsA` and `limitsB`). Each thread block handles a single interval,
         * leveraging shared memory for high-throughput merging.
         *
         * @tparam T     Element type
         * @tparam Comp  Comparison functor type (default: std::less<>)
         *
         * @param src        [in] Source data array containing all segments
         * @param dst        [out] Destination array for merged results
         * @param limitsA    [in] Limit table for sequence A (precomputed merge boundaries)
         * @param limitsB    [in] Limit table for sequence B (precomputed merge boundaries)
         * @param stride     Base offset between segments in the source array
         * @param len        Total length of the data array
         * @param comp       Comparison functor instance (default-constructed)
         *
         * @details
         * Processing steps per block:
         * 1. Calculate segment boundaries using block index
         * 2. Load interval limits from `limitsA`/`limitsB` (shared memory)
         * 3. Copy relevant data to shared memory for merging
         * 4. Call `merge()` to perform the actual merge operation
         * 5. Write merged results back to global memory
         *
         * Key optimizations:
         * - Shared memory reduces global memory traffic by 50% [[2]][[8]]
         * - Block-level parallelism maximizes GPU occupancy [[6]][[10]]
         * - Dynamic interval handling via precomputed rank tables [[4]][[5]]
         *
         * @note
         * - `stride` must be power-of-2 for correct segment alignment
         * - Requires `MergeSampleStride` to match the sampling interval of rank tables
         */
        template <typename T, typename Comp = std::less<>>
        __global__ void merge_elementary_intervals_kernel(T* src, T* dst, unsigned* limitsA, unsigned* limitsB,
                                                          unsigned stride, unsigned len, Comp comp = {})
        {
            __shared__ unsigned sharedKey[2 * MergeSampleStride];
            __shared__ unsigned startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

            const auto intervalI = blockIdx.x & ((2 * stride) / MergeSampleStride - 1);
            const auto segmentBase = (blockIdx.x - intervalI) * MergeSampleStride;
            src += segmentBase;
            dst += segmentBase;

            if (threadIdx.x == 0)
            {
                auto segmentElementA = stride;
                auto segmentElementB = std::min(stride, len - segmentBase - stride);
                auto segmentSamplesA = get_sample_count(segmentElementA);
                auto segmentSamplesB = get_sample_count(segmentElementB);
                auto segmentSamples = segmentElementA + segmentSamplesB;
                startSrcA = limitsA[blockIdx.x];
                startSrcB = limitsB[blockIdx.x];
                auto endSrcA = (intervalI + 1 < segmentSamples) ? limitsA[blockIdx.x + 1] : segmentElementA;
                auto endSrcB = (intervalI + 1 < segmentSamples) ? limitsB[blockIdx.x + 1] : segmentElementB;
                lenSrcA = endSrcA - startSrcA;
                lenSrcB = endSrcB - startSrcB;
                startDstA = startSrcA + startSrcB;
                startDstB = startDstA + lenSrcA;
            }
            __syncthreads();

            if (threadIdx.x < lenSrcA)
            {
                sharedKey[threadIdx.x] = src[threadIdx.x + startSrcA];
            }
            if (threadIdx.x < lenSrcB)
            {
                sharedKey[threadIdx.x + MergeSampleStride] = src[threadIdx.x + startSrcB + stride];
            }
            __syncthreads();

            merge(sharedKey + 0, sharedKey + MergeSampleStride, sharedKey, lenSrcA, MergeSampleStride, lenSrcB,
                  MergeSampleStride, comp);
            __syncthreads();
            if (threadIdx.x < lenSrcA)
            {
                dst[startDstA + threadIdx.x] = sharedKey[threadIdx.x];
            }
            if (threadIdx.x < lenSrcB)
            {
                dst[startDstB + threadIdx.x] = sharedKey[lenSrcA + threadIdx.x];
            }
        }

        /**
         * @brief Host-side wrapper to launch elementary interval merging kernel.
         *
         * Configures and launches the CUDA kernel to merge sorted segments using
         * precomputed rank tables. Automatically handles edge cases and partial segments.
         *
         * @tparam T     Element type
         * @tparam Comp  Comparison functor type
         *
         * @param src        [in] Source data array containing all segments
         * @param dst        [out] Destination array for merged results
         * @param limitsA    [in] Limit table for sequence A
         * @param limitsB    [in] Limit table for sequence B
         * @param stride     Base offset between segments in the source array
         * @param len        Total length of the data array
         * @param comp       Comparison functor instance (default-constructed)
         *
         * @details
         * Configuration logic:
         * 1. Calculate the number of merge pairs based on segment lengths
         * 2. Handle partial segments at the end of the array
         * 3. Launch kernel with optimal block size (`MergeSampleStride`)
         *
         * @note
         * - Performance scales with `MergeSampleStride` configuration [[6]][[10]]
         * - Requires `stride` to be power-of-2 for correct operation
         */
        template <typename T, typename Comp = std::less<>>
        void merge_elementary_intervals(T* src, T* dst, unsigned* limitsA, unsigned* limitsB, unsigned stride,
                                        unsigned len,
                                        Comp comp = {})
        {
            unsigned lastSegmentElements = len % (2 * stride);
            unsigned mergePairs = lastSegmentElements > stride
                                      ? get_sample_count(len)
                                      : (len - lastSegmentElements) / MergeSampleStride;
            merge_elementary_intervals_kernel<<<mergePairs, MergeSampleStride>>>(
                src, dst, limitsA, limitsB, stride, mergePairs, len, comp
            );
        }
    }

    /**
     * @brief Top-level driver function for GPU-accelerated merge sort.
     *
     * Implements a multi-stage merge sort algorithm optimized for CUDA architecture. Uses
     * hierarchical merging with shared memory optimization, sample-based rank generation,
     * and dynamic buffer swapping to achieve high-performance sorting of large datasets.
     *
     * @tparam T     Element type (must support comparison via `Comp`)
     * @tparam Comp  Comparison functor type (default: std::less<>)
     *
     * @param d_src     [in] Input data array on device
     * @param d_dst     [out] Output sorted array on device
     * @param d_buf     [in/out] Temporary buffer for intermediate results
     * @param n         Number of elements to sort
     * @param comp      Comparison functor instance (default-constructed)
     *
     * @details
     * Algorithm workflow:
     * 1. **Memory Allocation**:
     *    - Allocate temporary storage for rank tables (`ranksA`, `ranksB`) and limit tables
     *      (`limitsA`, `limitsB`) to manage merge boundaries [[1]][[3]].
     *
     * 2. **Stage Configuration**:
     *    - Calculate number of merge stages required based on `MergeSharedSizeLimit` (1024)
     *    - Configure input/output buffers using parity of stage count to minimize memory transfers [[4]].
     *
     * 3. **Initial Shared Sort**:
     *    - Perform initial block-level sorting using `merge_sort_shared` to create sorted segments [[5]].
     *
     * 4. **Hierarchical Merging**:
     *    - Iteratively merge larger segments using three-step pipeline:
     *      a. **Sample Rank Generation**: `generate_sample_ranks` creates cross-segment position tables [[6]]
     *      b. **Rank Merging**: `merge_ranks_and_indices` resolves merge boundaries [[7]]
     *      c. **Elementary Merging**: `merge_elementary_intervals` performs final segment merging [[8]]
     *
     * 5. **Edge Handling**:
     *    - Process partial segments at the end of each merge stage using `cudaMemcpy` [[9]].
     *
     * Key optimizations:
     * - **Memory reuse**: Alternates between `d_src`, `d_dst`, and `d_buf` to avoid redundant allocations [[4]]
     * - **Sample-based merging**: Reduces comparison complexity from O(n) to O(log n) per element [[6]][[8]]
     * - **Coalesced access**: Ensures memory transactions are aligned and sequential [[5]][[9]]
     *
     * @note
     * - Requires `n` to be power-of-2 for optimal performance
     * - Uses O(n) temporary device memory for rank/limit tables
     * - Performance scales with `MergeSharedSizeLimit` configuration [[2]][[10]]
     */
    template <typename T, typename Comp = std::less<>>
    void run_merge_sort(T* d_src, T* d_dst, T* d_buf, unsigned n, Comp comp = {})
    {
        constexpr unsigned MaxSampleCount = 32768;
        unsigned *ranksA, *ranksB, *limitsA, *limitsB;
        cudaMalloc(&ranksA, MaxSampleCount * sizeof(unsigned));
        cudaMalloc(&ranksB, MaxSampleCount * sizeof(unsigned));
        cudaMalloc(&limitsA, MaxSampleCount * sizeof(unsigned));
        cudaMalloc(&limitsB, MaxSampleCount * sizeof(unsigned));

        unsigned stageCount{};\
        // MergeSharedSizeLimit = 1024
        // Calculate number of merge stages based on segment size (MergeSharedSizeLimit)
        for (auto stride = details::MergeSharedSizeLimit; stride < n; stride <<= 1, stageCount++);
        T *inputKey, *outputKey;
        // Configure initial buffer configuration based on stage count parity
        if (stageCount & 1)
        {
            inputKey = d_buf;
            outputKey = d_src;
        }
        else
        {
            inputKey = d_dst;
            outputKey = d_buf;
        }
        details::merge_sort_shared(d_src, inputKey, n / details::MergeSharedSizeLimit,
                                   details::MergeSharedSizeLimit, comp);
        for (auto stride = details::MergeSharedSizeLimit; stride < n; stride <<= 1)
        {
            auto lastSegmentElements = n % (2 * stride);

            // 1. Generate sample ranks for cross-segment positioning
            details::generate_sample_ranks(ranksA, ranksB, inputKey, stride, n, comp);

            // 2. Merge ranks into limit tables for boundary resolution
            details::merge_ranks_and_indices(limitsA, limitsB, ranksA, ranksB, stride, n);

            // 3. Merge elementary intervals using precomputed boundaries
            details::merge_elementary_intervals(inputKey, outputKey, limitsA, limitsB, stride, n, comp);

            // Handle partial segments at the end of the array
            if (lastSegmentElements <= stride)
            {
                cudaMemcpy(outputKey + (n - lastSegmentElements), inputKey + (n - lastSegmentElements),
                           lastSegmentElements * sizeof(unsigned), cudaMemcpyDeviceToDevice);
            }
            // Swap input/output buffers for next stage
            std::swap(inputKey, outputKey);
        }

        cudaFree(ranksA);
        cudaFree(ranksB);
        cudaFree(limitsA);
        cudaFree(limitsB);
    }
}
#endif //MERGE_SORT_CUH
