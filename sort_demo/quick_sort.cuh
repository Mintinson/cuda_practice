//
// Created by asus on 2025/4/14.
//

#ifndef QUICK_SORT_CUH
#define QUICK_SORT_CUH

// #include <cooperative_groups.h>
#include <cstddef>
#include <functional>
#include <cuda_runtime.h>

namespace cuda_sort
{
    template <typename T, typename Compare = std::less<>>
    __device__ void selection_sort(T* data, int left, int right, Compare comp = {})
    {
        for (int i = left; i <= right; ++i)
        {
            T min_val = data[i];
            int min_idx = i;

            // Find the smallest value in the range [left, right].
            for (int j = i + 1; j <= right; ++j)
            {
                T val_j = data[j];

                if (comp(val_j, min_val))
                {
                    min_idx = j;
                    min_val = val_j;
                }
            }

            // Swap the values.
            if (i != min_idx)
            {
                data[min_idx] = data[i];
                data[i] = min_val;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Very basic quicksort algorithm, recursively launching the next level.
    ////////////////////////////////////////////////////////////////////////////////
    template <typename T, typename Compare = std::less<>>
    __global__ void cdp_simple_quicksort(T* data, int left, int right,
                                         int depth, Compare comp = {})
    {
        // If we're too deep or there are few elements left, we use an insertion
        // sort...
        constexpr int MAX_DEPTH = 32;
        constexpr int INSERTION_SORT = 32;
        if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
        {
            selection_sort(data, left, right);
            return;
        }

        T* lptr = data + left;
        T* rptr = data + right;
        T pivot = data[(left + right) / 2];

        // Do the partitioning.
        while (lptr <= rptr)
        {
            // Find the next left- and right-hand values to swap
            T lval = *lptr;
            T rval = *rptr;

            // Move the left pointer as long as the pointed element is smaller than the
            // pivot.
            while (comp(lval, pivot))
            {
                lptr++;
                lval = *lptr;
            }

            // Move the right pointer as long as the pointed element is larger than the
            // pivot.
            while (comp(pivot, rval))
            {
                rptr--;
                rval = *rptr;
            }

            // If the swap points are valid, do the swap!
            if (lptr <= rptr)
            {
                *lptr++ = rval;
                *rptr-- = lval;
            }
        }

        // Now the recursive part
        int nright = rptr - data;
        int nleft = lptr - data;

        // Launch a new block to sort the left part.
        if (left < (rptr - data))
        {
            cudaStream_t s;
            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            cdp_simple_quicksort<<<1, 1, 0, s>>>(data, left, nright, depth + 1);
            cudaStreamDestroy(s);
        }

        // Launch a new block to sort the right part.
        if ((lptr - data) < right)
        {
            cudaStream_t s1;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cdp_simple_quicksort<<<1, 1, 0, s1>>>(data, nleft, right, depth + 1);
            cudaStreamDestroy(s1);
        }
    }

    namespace details
    {
        constexpr std::size_t StackSize = 1024 * 1024; // Stack elem count must be power of 2!
        constexpr std::size_t BitonicSortLen = 1024; // Must be power of 2!
        constexpr std::size_t QSortMaxDepth = 16; // Will force final bitonic stage at depth QSortMaxDepth+1
        constexpr std::size_t QSortBlockSizeShift = 9;
        constexpr std::size_t QSortBlockSize = 1 << QSortBlockSizeShift;

        struct __align__(128) QSortAtomicData
        {
            volatile unsigned ltOffset; // Current output offset for <pivot
            volatile unsigned gtOffset; // Current output offset for >pivot
            volatile unsigned sortedCount; // Total count sorted, for deciding when to launch next wave
            volatile unsigned index; // Ringbuf tracking index. Can be ignored if not using ringbuf.
        };

        struct QSortRingBuf
        {
            volatile unsigned int head; // Head pointer - we allocate from here
            volatile unsigned int tail; // Tail pointer - indicates last still-in-use element
            volatile unsigned int count; // Total count allocated
            volatile unsigned int max; // Max index allocated
            unsigned int stackSize; // Wrap-around size of buffer (must be power of 2)
            volatile void* stackBase; // Pointer to the stack we're allocating from
        };

        /**
         * @brief  Inline PTX call to return index of highest non-zero bit in a word
         * @param word
         * @return
         */
        static __device__ __forceinline__ unsigned int btflo(unsigned int word)
        {
            unsigned int ret;
            asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
            return ret;
        }

        static __device__ __forceinline__ unsigned int quick_sflo(unsigned word)
        {
            unsigned int ret{};
            asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
            return ret;
        }

        __device__ __forceinline__ int quick_compare(unsigned& val1, unsigned& val2)
        {
            return (val1 > val2)
                       ? 1
                       : val1 == val2
                       ? 0
                       : -1;
        }


        /**
         * @brief Allocates from a ringbuffer. Allows for not failing when we run out
         * of stack for tracking the offset counts for each sort subsection.
         *
         *  We use the atomicMax trick to allow out-of-order retirement. If we
         *  hit the size limit on the ringbuffer, then we spin-wait for people
         *  to complete.
         * @tparam T
         * @param ringBuf
         * @return
         */
        template <typename T>
        static __device__ T* ring_buf_allocate(QSortRingBuf* ringBuf)
        {
            // Wait for there to be space in the ring buffer. We'll retry only a fixed
            // number of times and then fail, to avoid an out-of-memory deadlock.
            unsigned loop = 10000;
            while (((ringBuf->head - ringBuf->tail) >= ringBuf->stackSize) && (loop-- > 0));

            if (loop == 0)
            {
                return nullptr;
            }
            // Note that the element includes a little index bookkeeping, for freeing
            // later.
            unsigned int index = atomicAdd(const_cast<unsigned int*>(&ringBuf->head), 1);
            T* ret = (T*)(ringBuf->stackBase) + (index & (ringBuf->stackSize - 1)); // index % stackSize
            ret->index = index;
            return ret;
        }

        /**
         * @brief Releases an element from the ring buffer. If every element is released
         * up to and including this one, we can advance the tail to indicate that
         * space is now available.
         * @tparam T
         * @param ringBuf
         * @param data
         */
        template <typename T>
        static __device__ void ring_buf_free(QSortRingBuf* ringBuf, T* data)
        {
            unsigned int index = data->index;
            unsigned int count = atomicAdd(const_cast<unsigned int*>(&ringBuf->count), 1) + 1;
            unsigned int maxV = atomicMax(const_cast<unsigned int*>(&ringBuf->max), index + 1);

            // Update the tail if need be. Note we update "max" to be the new value in
            // ringBuf->max
            if (maxV < (index + 1))
                maxV = index + 1;
            if (maxV == count)
                atomicMax(const_cast<unsigned int*>(&ringBuf->tail), count);
        }

        template <typename T, typename Comp = std::less<>>
        __global__ void bitonic_sort_kernel(const T* inData, T* outData,
                                            const unsigned offset, const std::size_t len,
                                            Comp comp = {})
        {
            // Max of 1024 elements - TODO: make this dynamic
            __shared__ T sortBuf[1024];
            auto tid = threadIdx.x;
            bool inside = threadIdx.x < len;
            sortBuf[tid] = inside
                               ? inData[tid + offset]
                               : comp(T{}, T{} + 1)
                               ? std::numeric_limits<T>::max()
                               : std::numeric_limits<T>::lowest();
            __syncthreads();

            // Now the sort loops
            // Here, "k" is the sort level (remember bitonic does a multi-level butterfly
            // style sort)
            // and "j" is the partner element in the butterfly.
            // Two threads each work on one butterfly, because the read/write needs to
            // happen
            // simultaneously
            for (unsigned int k = 2; k <= blockDim.x; k *= 2)
            {
                for (unsigned int j = k >> 1; j > 0; j >>= 1)
                {
                    auto swapIdx = tid ^ j;
                    auto myElem = sortBuf[tid];
                    auto swapVal = sortBuf[swapIdx];
                    __syncthreads();

                    // The k'th bit of my threadid (and hence my sort item ID)
                    // determines if we sort ascending or descending.
                    // However, since threads are reading from the top AND the bottom of
                    // the butterfly, if my ID is > swap_idx, then ascending means mine<swap.
                    // Finally, if either my_elem or swap_elem is out of range, then it
                    // ALWAYS acts like it's the largest number.
                    // Confusing? It saves us two writes though.
                    auto ascend = k * (swapIdx < tid);
                    auto descent = k * (swapIdx > tid);
                    bool swap = false;
                    if ((tid & k) == ascend)
                        if (comp(swapVal, myElem))
                            swap = true;
                    if ((tid & k) == descent)
                        if (comp(myElem, swapVal))
                            swap = true;
                    // If we had to swap, then write my data to the other element's position.
                    // Don't forget to track out-of-range status too!
                    if (swap)
                    {
                        sortBuf[swapIdx] = myElem;
                    }
                    __syncthreads();
                }
            }
            if (tid < len)
            {
                // Copy the sorted data from shared memory back to the output buffer
                outData[offset + tid] = sortBuf[tid];
            }
        }

        /**
         *  This is an emergency-CTA sort, which sorts an arbitrary sized chunk
         *  using a single block. Useful for if qsort runs out of nesting depth.
         *
         *  Note that bitonic sort needs enough storage to pad up to the nearest
         *  power of 2. This means that the double-buffer is always large enough
         *  (when combined with the main buffer), but we do not get enough space
         *  to keep OOR information.
         *
         *  This in turn means that this sort does not work with a generic data
         *  type. It must be a directly-comparable (i.e. with max value) type.
        */
        template <typename T, typename Comp = std::less<>>
        __global__ void big_bitonic_sort_kernel(T* inData, T* outData, T* backBuf,
                                                const unsigned int offset, const std::size_t len,
                                                Comp comp = {})
        {
            // Round up len to nearest power-of-2
            unsigned int len2 = 1 << (btflo(len - 1U) + 1);
            auto tid = threadIdx.x;
            // Early out for case where more threads launched than there is
            // data
            if (tid >= len2)
                return;
            const auto PaddingVal = comp(T{}, T{} + 1)
                                        ? std::numeric_limits<T>::max()
                                        : std::numeric_limits<T>::lowest();
            // First, set up our unused values to be the max data type.
            for (unsigned i = len; i < len2; i += blockDim.x)
            {
                unsigned int idx = i + tid;
                if (idx < len2)
                {
                    // Must split our index between two buffers
                    if (idx < len)
                        inData[idx + offset] = PaddingVal;
                    else
                        backBuf[idx + offset - len] = PaddingVal;
                }
            }
            __syncthreads();
            // Now the sort loops
            // Here, "k" is the sort level (remember bitonic does a multi-level butterfly
            // style sort)
            // and "j" is the partner element in the butterfly.
            // Two threads each work on one butterfly, because the read/write needs to
            // happen
            // simultaneously
            for (unsigned k = 2; k <= len2; k <<= 1)
            {
                for (unsigned j = k >> 1; j > 0; j >>= 1)
                {
                    for (unsigned i = 0; i < len2; i += blockDim.x)
                    {
                        auto idx = tid + i;
                        auto swapIdx = idx ^ j;
                        // Only do the swap for index<swap_idx (avoids collision between other
                        // threads)
                        if (swapIdx > idx)
                        {
                            auto myElem = idx < len ? inData[idx + offset] : backBuf[idx + offset - len];
                            auto swapVal = swapIdx < len
                                               ? inData[swapIdx + offset]
                                               : backBuf[swapIdx + offset - len];

                            // The k'th bit of my index (and hence my sort item ID)
                            // determines if we sort ascending or descending.
                            // Also, if either my_elem or swap_elem is out of range, then it
                            // ALWAYS acts like it's the largest number.
                            bool swap = false;

                            if ((idx & k) == 0 && comp(swapVal, myElem))
                                swap = true;
                            if ((idx & k) == k && comp(myElem, swapVal))
                                swap = true;
                            // If we had to swap, then write my data to the other element's
                            // position.
                            if (swap)
                            {
                                if (swapIdx < len)
                                    inData[swapIdx + offset] = myElem;
                                else
                                    backBuf[swapIdx + offset - len] = myElem;
                                if (idx < len)
                                    inData[idx + offset] = swapVal;
                                else
                                    backBuf[idx + offset - len] = swapVal;
                            }
                        }
                    }
                    __syncthreads();
                }
            }
            // Copy the sorted data from the input to the output buffer, because we sort
            // in-place
            if (outData != inData)
            {
                for (unsigned i = 0; i < len; i += blockDim.x)
                {
                    unsigned int idx = i + tid;
                    if (idx < len)
                        outData[idx + offset] = inData[idx + offset];
                }
            }
        }

        /**
         * @brief CUDA kernel implementing a warp-based parallel quicksort algorithm with custom comparator.
         *
         * This is a generic implementation of the warp-synchronized quicksort partitioning step.
         * It uses atomic operations, warp-level primitives, and dynamic parallelism to coordinate sorting,
         * with support for user-defined comparison logic and automatic fallback to bitonic sort for small partitions.
         *
         * @tparam T        Data type to be sorted (must support comparison via Compare)
         * @tparam Compare  Comparison functor (default: std::less<>)
         *
         * @param inData           [in] Input data array (current pass input)
         * @param outData          [out] Output data array (current pass output)
         * @param offset           Starting index of the current sub-array
         * @param len              Length of the current sub-array
         * @param atomicData       Atomic counter structure for partition coordination
         * @param atomicDataStack  Ring buffer for managing dynamic memory allocations
         * @param sourceIsInData   Flag indicating if source data is in inData (0=outData, 1=inData)
         * @param depth            Current recursion depth (to prevent stack overflow)
         * @param comp             Comparison functor instance (default-constructed)
         *
         * @details
         * Algorithm workflow:
         * 1. Warp selects pivot (mid-element of current sub-array)
         * 2. Threads collaboratively count elements <=/> pivot using warp vote (__ballot_sync)
         * 3. Atomic operations update global offsets for left/right partitions
         * 4. Threads compute their destination position in output array
         * 5. Last completing warp triggers:
         *    - Recursive sorting of sub-partitions (if length > BitonicSortLen)
         *    - Bitonic sort for small partitions (<= BitonicSortLen)
         *    - Fallback to bitonic sort if recursion depth exceeds QSortMaxDepth
         *
         * Key CUDA features used:
         * - Warp-level primitives (__ballot_sync, __popc, __shfl_sync)
         * - Dynamic parallelism (kernel launches from device)
         * - Asynchronous streams for concurrent partition sorting
         * - Ring buffer memory management for atomic data
         *
         * Edge Cases Handled:
         * - All elements equal (ltLen == 0)
         * - Single-element partitions (ltLen == 1)
         * - Maximum recursion depth protection
         *
         * @note Requires:
         * - Compute capability 7.0+ for warp-level functions
         * - Block size matching QSortBlockSize
         * - Properly initialized atomicDataStack
         *
         * @warning
         * - Exceeding QSortMaxDepth will fallback to bitonic sort
         * - Recursive kernel launches use non-blocking streams
         */
        template <typename T, typename Compare = std::less<>>
        __global__ void quick_sort_warp(T* inData, T* outData, unsigned int offset,
                                        unsigned int len,
                                        QSortAtomicData* atomicData, QSortRingBuf* atomicDataStack,
                                        unsigned int sourceIsInData, unsigned int depth,
                                        Compare comp = {})
        {
            // Find my data offset, based on warp ID
            auto tid = threadIdx.x;
            auto idx = tid + (blockIdx.x << details::QSortBlockSizeShift);
            auto laneId = tid & (warpSize - 1);

            // Exit if I'm outside the range of sort to be done
            if (idx >= len)
                return;

            // First part of the algorithm. Each warp counts the number of elements that
            // are greater/less than the pivot.
            // When a warp knows its count, it updates an atomic counter.
            auto pivot = inData[offset + len / 2];
            auto data = inData[offset + idx];

            // Count how many are <= and how many are > pivot.
            // If all are <= pivot then we adjust the comparison
            // because otherwise the sort will move nothing and
            // we'll iterate forever.

            unsigned greater = comp(pivot, data);
            auto gtMask = __ballot_sync(0xffffffff, greater);
            // auto gtMask = ballot(greater);

            if (gtMask == 0)
            {
                greater = !comp(data, pivot);
                gtMask = __ballot_sync(0xffffffff, greater); // Must re-ballot for adjusted comparator
            }
            auto ltMask = __ballot_sync(0xffffffff, !greater);
            // __popc: Count the number of bits that are set to 1 in a 32-bit integer.
            unsigned gtCount = __popc(gtMask);
            unsigned ltCount = __popc(ltMask);

            // Atomically adjust the lt_ and gtOffsets by this amount. Only one thread
            // need do this. Share the result using shfl
            unsigned ltOffset{};
            unsigned gtOffset{};
            if (laneId == 0)
            {
                if (ltCount > 0)
                    ltOffset = atomicAdd(const_cast<unsigned int*>(&(atomicData->ltOffset)), ltCount);
                if (gtCount > 0)
                    gtOffset =
                        len - (atomicAdd(const_cast<unsigned int*>(&atomicData->gtOffset), gtCount) +
                            gtCount);
            }
            // Everyone pulls the offsets from lane 0
            ltOffset = __shfl_sync(0xfffffff, static_cast<int>(ltOffset), 0);
            gtOffset = __shfl_sync(0xfffffff, static_cast<int>(gtOffset), 0);
            __syncthreads();

            // Now compute my own personal offset within this. I need to know how many
            // threads with a lane ID less than mine are going to write to the same buffer
            // as me. We can use popc to implement a single-operation warp scan in this
            // case.
            unsigned lane_mask_lt;
            asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
            unsigned int myMask = greater ? gtMask : ltMask;
            unsigned int myOffset = __popc(myMask & lane_mask_lt);

            // Move data.
            myOffset += greater ? gtOffset : ltOffset;
            outData[offset + myOffset] = data;

            // Count up if we're the last warp in. If so, then Kepler will launch the next
            // set of sorts directly from here.
            if (laneId == 0)
            {
                // Count "elements written". If I wrote the last one, then trigger the next
                // qsorts
                auto myCount = ltCount + gtCount;
                if (atomicAdd(const_cast<unsigned int*>(&(atomicData->sortedCount)), myCount) + myCount == len)
                {
                    // We're the last warp to do any sorting. Therefore, it's up to us to
                    // launch the next stage.
                    auto ltLen = atomicData->ltOffset;
                    auto gtLen = atomicData->gtOffset;

                    cudaStream_t lStream;
                    cudaStream_t rStream;
                    cudaStreamCreateWithFlags(&lStream, cudaStreamNonBlocking);
                    cudaStreamCreateWithFlags(&rStream, cudaStreamNonBlocking);

                    // Begin by freeing our atomicData storage. It's better for the ringbuffer
                    // algorithm
                    // if we free when we're done, rather than re-using (makes for less
                    // fragmentation).
                    ring_buf_free<QSortAtomicData>(atomicDataStack, atomicData);

                    // Exceptional case: if "ltLen" is zero, then all values in the batch
                    // are equal. We are then done (may need to copy into correct buffer,
                    // though)
                    if (ltLen == 0)
                    {
                        if (sourceIsInData)
                            cudaMemcpyAsync(inData + offset, outData + offset, gtLen * sizeof(unsigned),
                                            cudaMemcpyDeviceToDevice, lStream);
                        return;
                    }
                    // Start with lower half first
                    if (ltLen > BitonicSortLen)
                    {
                        // If we've exceeded maximum depth, fall through to back up
                        // big_bitonicsort
                        if (depth >= QSortMaxDepth)
                        {
                            // The final bitonic stage sorts in-place in "outData". We therefore
                            // re-use "inData" as the out-of-range tracking buffer. For (2^n)+1
                            // elements we need (2^(n+1)) bytes of our buffer. The backup qsort
                            // buffer is at least this large when sizeof(T) >= 2.
                            big_bitonic_sort_kernel<<<1, BitonicSortLen, 0, lStream>>>(
                                outData, sourceIsInData ? inData : outData,
                                inData, offset, ltLen, comp);
                        }
                        else
                        {
                            // Launch another quicksort. We need to allocate more storage for the
                            // atomic data.
                            if ((atomicData = ring_buf_allocate<QSortAtomicData>(atomicDataStack)) ==
                                nullptr)
                            {
                                printf("Stack-allocation error. Failing left child launch.\n");
                            }
                            else
                            {
                                atomicData->ltOffset = atomicData->gtOffset = atomicData->sortedCount = 0;
                                unsigned int numBlocks =
                                    static_cast<unsigned int>(ltLen + (QSortBlockSize - 1)) >>
                                    QSortBlockSizeShift;
                                quick_sort_warp<<<numBlocks, QSortBlockSize, 0, lStream>>>(
                                    outData, inData, offset, ltLen, atomicData, atomicDataStack,
                                    !sourceIsInData, depth + 1, comp);
                            }
                        }
                    }
                    else if (ltLen > 1)
                    {
                        // Final stage uses a bitonic sort instead. It's important to
                        // make sure the final stage ends up in the correct (original) buffer.
                        // We launch the smallest power-of-2 number of threads that we can.
                        unsigned bitonicLen = 1 << (quick_sflo(ltLen - 1U) + 1);
                        bitonic_sort_kernel<<<1, bitonicLen, 0, lStream>>>(
                            outData, sourceIsInData ? inData : outData, offset, ltLen, comp);
                    }
                    // Finally, if we sorted just one single element, we must still make
                    // sure that it winds up in the correct place.
                    else if (sourceIsInData && (ltLen == 1))
                    {
                        inData[offset] = outData[offset];
                    }

                    // Now the upper half.
                    if (gtLen > BitonicSortLen)
                    {
                        // If we've exceeded maximum depth, fall through to backup
                        // big_bitonicsort
                        if (depth >= QSortMaxDepth)
                        {
                            big_bitonic_sort_kernel<<<1, BitonicSortLen, 0, rStream>>>(
                                outData, sourceIsInData ? inData : outData, inData, offset + ltLen,
                                gtLen, comp);
                        }
                        else
                        {
                            if ((atomicData = ring_buf_allocate<QSortAtomicData>(atomicDataStack)) ==
                                nullptr)
                            {
                                printf("stack allocation error! Failing right-side launch\n");
                            }
                            else
                            {
                                atomicData->ltOffset = atomicData->gtOffset = atomicData->sortedCount = 0;
                                unsigned int numBlocks =
                                    static_cast<unsigned int>(gtLen + (QSortBlockSize - 1)) >>
                                    QSortBlockSizeShift;
                                quick_sort_warp<<<numBlocks, QSortBlockSize, 0, rStream>>>(
                                    outData, inData, offset + ltLen, gtLen, atomicData, atomicDataStack,
                                    !sourceIsInData, depth + 1, comp);
                            }
                        }
                    }
                    else if (gtLen > 1)
                    {
                        unsigned bitonicLen = 1 << (quick_sflo(gtLen - 1U) + 1);
                        bitonic_sort_kernel<<<1, bitonicLen, 0, rStream>>>(
                            outData, sourceIsInData ? inData : outData, offset + ltLen, gtLen, comp);
                    }
                    else if (sourceIsInData && (gtLen == 1))
                    {
                        inData[offset + ltLen] = outData[offset + ltLen];
                    }
                }
            }
        }
    }

    template <typename T, typename Compare = std::less<>>
    void run_quick_sort_cdp(T* d_a, T* d_buffer, std::size_t n, Compare comp = {})
    {
        std::size_t stackSize = details::StackSize;
        details::QSortAtomicData* gpuStack;
        cudaMalloc(reinterpret_cast<void**>(&gpuStack), stackSize * sizeof(details::QSortAtomicData));
        cudaMemset(gpuStack, 0, sizeof(details::QSortAtomicData) * stackSize);

        // Create the memory ringbuffer used for handling the stack.
        // Initialise everything to where it needs to be.
        details::QSortRingBuf buf;
        details::QSortRingBuf* ringBuf;
        cudaMalloc(reinterpret_cast<void**>(&ringBuf), sizeof(details::QSortRingBuf));
        buf.head = 1; // We start with one allocation
        buf.tail = 0;
        buf.count = 0;
        buf.max = 0;
        buf.stackSize = stackSize;
        buf.stackBase = gpuStack;

        cudaMemcpy(ringBuf, &buf, sizeof(buf), cudaMemcpyHostToDevice);
        if (n > details::BitonicSortLen)
        {
            unsigned int numBlocks =
                static_cast<unsigned int>(n + (details::QSortBlockSize - 1)) >> details::QSortBlockSizeShift;

            details::quick_sort_warp<<<numBlocks, details::QSortBlockSize>>>(
                d_a, d_buffer, 0U, n, gpuStack, ringBuf, true, 0, comp);
        }
        else
        {
            details::bitonic_sort_kernel<<<1, details::BitonicSortLen>>>(d_a, d_a, 0, n, comp);
        }
        cudaFree(ringBuf);
        cudaFree(gpuStack);
    }
}

#endif // QUICK_SORT_CUH
