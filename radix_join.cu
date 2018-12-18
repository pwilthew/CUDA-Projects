/*
    Programming on Massively Parallel Systems
    Fall 2018
    Project # 3
    Student: Patricia Wilthew

    Compile: nvcc proj3.cu -o proj3
    Usage: ./proj3 {#of_elements_in_array1} {#of_elements_in_array2}
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <assert.h>
#include <thrust/scan.h>
#define RAND_RANGE(N) ((double)rand()/((double)RAND_MAX + 1)*(N))
#define PARTITION_LENGTH 128
#define SECTION_LENGTH 4

cudaError_t err;
__device__ int matches = 0;

/*
    Function: catch_error

    Description: Prints any CUDA error to stdout.
*/
void catch_error(cudaError_t error)
{
    if (error)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}

/*
    Function: data_generator

    Description: Uses Knuth Shuffle to generate integers in data.
*/
void data_generator(int* data, int count, int first, int step)
{
    assert(data != NULL);

    for (int i = 0; i < count; ++i)
    {
        data[i] = first + i * step;
    }
    
    srand(time(NULL));
    
    for (int i = count-1; i>0; i--) //knuth shuffle
    {
        int j = RAND_RANGE(i);
        int k_tmp = data[i];
        data[i] = data[j];
        data[j] = k_tmp;
    }
}

/*
    Function: bfe

    Description: This function embeds PTX code of CUDA to extract bit
        field from x.

    Input:
        start (uint): Starting bit position relative to the LSB.
        nbits (uint): The bit field length.

    Output: The extracted bit field as an unsigned integer.
*/
__device__ uint bfe(uint x, uint start, uint nbits)
{
    uint bits;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(x), "r"(start), "r"(nbits));
    return bits;
}

/*
    Function: histogram

    Description: In order to efficiently partition the input in parallel,
        first we need to compute a histogram of the radix values by
        scanning the array of keys so that we know the number of keys
        that should go to each partition.
*/
__global__ void histogram(int *array, int array_length, int *hist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radix;

    if (idx < array_length)
    {
        radix = (int) bfe(array[idx], 0, 31-__builtin_clz(PARTITION_LENGTH));
        atomicAdd(&hist[blockIdx.x * PARTITION_LENGTH + radix], 1);
    }
}

/*
    Function: organize_histogram

    Description: Given a histogram `hist` of the form:
        [w1, x1, y1, z1,   w2, x2, y2, z2,   w3, x3, y3, z3], create an
        `organized` histogram of the form:
        [w1, w2, w3,   x1, x2, x3,   y1, y2, y3,   z1, z2, z3].
*/
__global__ void organize_histogram(int *hist, int hist_length, int *organized)
{
    int number_of_sections = hist_length / PARTITION_LENGTH;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value, section_offset;

    if (idx < hist_length)
    {
        section_offset = idx/number_of_sections;
        value = hist[
                    PARTITION_LENGTH * (idx % number_of_sections)
                    + section_offset
                ];
        organized[idx] = value;
    }
}

/*
    Function: Kogge_Stone_scan_kernel

    Description: This is the first of the three kernels needed to implement the
        hierarchical scan with three kernels. It uses Kogge Stone scan
        algorithm to compute the prefix sum of each block of length
        SECTION_LENGTH.
*/
__global__ void Kogge_Stone_block_scan_kernel(int *X, int *Y, int *S, int length)
{
    __shared__ int XY[SECTION_LENGTH];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length)
    {
        XY[threadIdx.x] = X[idx];
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x-stride];
    }

    Y[idx] = XY[threadIdx.x];

    __syncthreads();
    
    // If this is the last thread of the block:
    if (threadIdx.x == blockDim.x - 1)
    {
       S[blockIdx.x] = XY[SECTION_LENGTH - 1];
    }
}

/*
    Function: parallel_scan_kernel

    Description: The second kernel of the hierarchical scan is simply one of the
        three parallel scan kernels, which takes S as input and writes S as
        output.

*/
__global__ void parallel_scan_kernel(int *X, int *Y, int length)
{
    __shared__ int XY[SECTION_LENGTH];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length)
    {
        XY[threadIdx.x] = X[idx];
    }

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride) XY[threadIdx.x] += XY[threadIdx.x-stride];
    }

    Y[idx] = XY[threadIdx.x];
}

/*
    Function: add_S_to_Y_kernel

    Description: The third kernel of the hierarchical scan takes the S and Y
        arrays as inputs and writes its output back into Y. Assuming that we
        launch the kernel with SECTION_LENGTH threads in each block, each thread
        adds one of the S elements (selected by blockIdx.x-1) to one Y element.
*/
__global__ void add_S_to_Y_kernel(int *Y, int *S, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int j, sum;

    if (blockIdx.x > 0 && idx < length)
    {
        Y[idx] += S[blockIdx.x - 1];
    
        // Also add the prefix sum of previous sections.
        if (blockIdx.x >= SECTION_LENGTH)
        {
            sum = 0;
            for (j = SECTION_LENGTH - 1; j < blockIdx.x - 1; j += 4)
            {
                sum += S[j];
            }
            Y[idx] += sum;
        }
    }
}

/* 
    Function: shift_right

    Description: Performs 1-shift right on original array and stores it in a new
        array. This is performed to convert an inclusive prefix sum into
        exclusive.
*/
__global__ void shift_right(int *original, int *shifted)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0)
    {
        shifted[idx] = 0;
    }
    else
    {
        shifted[idx] = original[idx - 1];
    }
}

/*
    Function: reduce_prefix_sum

    Description: Given a `prefix_sum` array of the form:
        [0, 6, 15, 24, 30, 37, 44, 51, 60, 71, 79, 87, 90, 98, 106, 114],
        Create a `reduced` array of PARTITION_LENGTH length, 4, of the form:
        [0, 30, 60, 90] that contains the index in which each partition starts.
*/
__global__ void reduce_prefix_sum(int *prefix_sum, int *reduced, int *reduced_copy, int p_length)
{
    int number;
    number = prefix_sum[blockIdx.x * p_length];
    reduced[blockIdx.x] = number;
    reduced_copy[blockIdx.x] = number;
}

/*
    Function: re_order

    Description: Re-order original `array` by radix into a partitioned new
        array, `ordered`.
*/
__global__ void re_order(int *prefix_sum, int *array, int *ordered, int length)
{
    int offset;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radix = (int) bfe(array[idx], 0, 31-__builtin_clz(PARTITION_LENGTH));

    if (idx < length)
    {
        // `offset` will have the value of prefix_sum[radix]
        // before it is incremented by one.
        offset = atomicAdd(&prefix_sum[radix], 1);
        ordered[offset] = array[idx];
    }
}

/*
    Function: probe

    Description: With the reordered keys from both input arrays, we can
        now perform the probe stage by only comparing the keys from the
        corresponding partitions of both input arrays. This can be done
        by nested-loop comparisons.
*/
__global__ void probe(int *r_delim, int *s_delim, int *r, int *s, int r_length, int s_length)
{
    int r_start, r_end, s_start, s_end;
    int i, j, count = 0;

    // Get delimiters of current partition.
    r_start = r_delim[blockIdx.x];
    s_start = s_delim[blockIdx.x];

    if (blockIdx.x == PARTITION_LENGTH - 1)
    {
        r_end = r_length - 1;
        s_end = s_length - 1;
    }
    else
    {
        r_end = r_delim[blockIdx.x + 1] - 1;
        s_end = s_delim[blockIdx.x + 1] - 1;
    }
    
    for (i=r_start; i <= r_end; i++)
    {
        for (j=s_start; j <= s_end; j++)
        {
            if (r[i] == s[j])
            {   
                count += 1;
            }
        }
    }
    atomicAdd(&matches, count);

}

__global__ void print_matches()
{
    printf("---> %d matches\n", matches);
}


int main(int argc, char const *argv[])
{
    if (argc <= 2)
    {
        printf("Usage: ./proj3.out {#of_elements_in_array1} {#of_elements_in_array2}\n");
        exit(1);
    }

    int r_length = atoi(argv[1]);
    int s_length = atoi(argv[2]);
    int r_size = sizeof(int)*r_length;
    int s_size = sizeof(int)*s_length;
    int r_hist_len, r_hist_size;
    int s_hist_len, s_hist_size;
    int blocks;
    double threads = 1024.0;
    float time, total_time = 0.0;

    int *r_host, *r_hist, *r_prefix_sum, *r_ordered, *r_reduced_prefix, 
        *r_reduced_prefix_copy;
    int *s_host, *s_hist, *s_prefix_sum, *s_ordered, *s_reduced_prefix,
        *s_reduced_prefix_copy;
    
    int *X, *Y, *S, *S2;
    
    // Allocate arrays in host and device.
    cudaMallocHost((void**)&r_host, r_size);
    cudaMallocHost((void**)&s_host, s_size);

    // Populate arrays.
    data_generator(r_host, r_length, 0, 1);
    data_generator(s_host, s_length, 0, 1);

    // Recording variables.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    /************************************
    *                                   *
    *   Order r into partitioned array  *
    *                                   *
    ************************************/
    blocks = ceil(r_length/threads);
    r_hist_len = PARTITION_LENGTH * blocks;
    r_hist_size = sizeof(int)*r_hist_len;
    cudaMalloc((void**)&r_hist, r_hist_size); cudaMemset(r_hist, 0, r_hist_size);
    cudaMalloc((void**)&X, r_hist_size);
    cudaMalloc((void**)&Y, r_hist_size);
    cudaMalloc((void**)&S, r_hist_size/SECTION_LENGTH);
    cudaMalloc((void**)&S2, r_hist_size/SECTION_LENGTH);
    cudaMalloc((void**)&r_prefix_sum, r_hist_size);
    cudaMalloc((void**)&r_reduced_prefix, PARTITION_LENGTH*sizeof(int));
    cudaMalloc((void**)&r_reduced_prefix_copy, PARTITION_LENGTH*sizeof(int));
    cudaMalloc((void**)&r_ordered, r_size); cudaMemset(r_ordered, 0, r_size);

    // Histogram for r: r_hist.
    cudaEventRecord(start, 0);
    histogram<<<blocks, threads>>>(r_host, r_length, r_hist);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Organized histogram for r: X
    cudaEventRecord(start, 0);
    organize_histogram<<<ceil(r_hist_len/threads), threads>>>(
        r_hist, r_hist_len, X);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for r (Part I): Y and S.
    cudaEventRecord(start, 0);
    Kogge_Stone_block_scan_kernel<<<
        ceil(r_hist_len/(SECTION_LENGTH)),
        SECTION_LENGTH>>>(X, Y, S, r_hist_len);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for r (Part II): S2.
    cudaEventRecord(start, 0);
    parallel_scan_kernel<<<ceil(r_hist_len/SECTION_LENGTH), SECTION_LENGTH>>>(
        S, S2, r_hist_len/SECTION_LENGTH);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for r (Part III): Y.
    cudaEventRecord(start, 0);
    add_S_to_Y_kernel<<<
        ceil(r_hist_len/SECTION_LENGTH),
        SECTION_LENGTH>>>(Y, S2, r_hist_len);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Convert to exclusive prefix sum.
    cudaEventRecord(start, 0);
    shift_right<<<
        ceil(r_hist_len/SECTION_LENGTH),
        SECTION_LENGTH>>>(Y, r_prefix_sum);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Obtain reduced prefix sum (of partition size)
    cudaEventRecord(start, 0);
    reduce_prefix_sum<<<PARTITION_LENGTH, 1>>>(
        r_prefix_sum, r_reduced_prefix, r_reduced_prefix_copy, blocks);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Reorder R array.
    cudaEventRecord(start, 0);
    re_order<<<ceil(r_length/threads), threads>>>(
        r_reduced_prefix, r_host, r_ordered, r_length);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;


    /************************************
    *                                   *
    *   Order s into partitioned array  *
    *                                   *
    ************************************/
    blocks = ceil(s_length/threads);
    s_hist_len = PARTITION_LENGTH * blocks;
    s_hist_size = sizeof(int)*s_hist_len;
    cudaMalloc((void**)&s_hist, s_hist_size); cudaMemset(s_hist, 0, s_hist_size);
    cudaMalloc((void**)&X, s_hist_size);
    cudaMalloc((void**)&Y, s_hist_size);
    cudaMalloc((void**)&S, s_hist_size/SECTION_LENGTH);
    cudaMalloc((void**)&S2, s_hist_size/SECTION_LENGTH);
    cudaMalloc((void**)&s_prefix_sum, s_hist_size);
    cudaMalloc((void**)&s_reduced_prefix, PARTITION_LENGTH*sizeof(int));
    cudaMalloc((void**)&s_reduced_prefix_copy, PARTITION_LENGTH*sizeof(int));
    cudaMalloc((void**)&s_ordered, s_size); cudaMemset(s_ordered, 0, s_size);

    // Histogram for s: s_hist.
    cudaEventRecord(start, 0);
    histogram<<<blocks, threads>>>(s_host, s_length, s_hist);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Organized histogram for s: X
    cudaEventRecord(start, 0);
    organize_histogram<<<ceil(s_hist_len/threads), threads>>>(
        s_hist, s_hist_len, X);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for s (Part I): Y and S.
    cudaEventRecord(start, 0);
    Kogge_Stone_block_scan_kernel<<<
        ceil(s_hist_len/(SECTION_LENGTH)),
        SECTION_LENGTH>>>(X, Y, S, s_hist_len);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for s (Part II): S2.
    cudaEventRecord(start, 0);
    parallel_scan_kernel<<<ceil(s_hist_len/SECTION_LENGTH), SECTION_LENGTH>>>(
        S, S2, s_hist_len/SECTION_LENGTH);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Hierarchical scan for s (Part III): Y.
    cudaEventRecord(start, 0);
    add_S_to_Y_kernel<<<
        ceil(s_hist_len/SECTION_LENGTH),
        SECTION_LENGTH>>>(Y, S2, s_hist_len);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Convert to exclusive prefix sum.
    cudaEventRecord(start, 0);
    shift_right<<<
        ceil(s_hist_len/SECTION_LENGTH),
        SECTION_LENGTH>>>(Y, s_prefix_sum);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Obtain reduced prefix sum (of partition size)
    cudaEventRecord(start, 0);
    reduce_prefix_sum<<<PARTITION_LENGTH, 1>>>(
        s_prefix_sum, s_reduced_prefix, s_reduced_prefix_copy, blocks);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    // Reorder s array.
    cudaEventRecord(start, 0);
    re_order<<<
        ceil(s_length/threads),
        threads>>>(s_reduced_prefix, s_host, s_ordered, s_length);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;


    /************************************
    *                                   *
    *               Probing             *
    *                                   *
    ************************************/
    cudaEventRecord(start, 0);
    probe<<<PARTITION_LENGTH, 1>>>(
        r_reduced_prefix_copy, s_reduced_prefix_copy, r_ordered, s_ordered,
        r_length, s_length);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    total_time += time;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaDeviceSynchronize();
    printf(
        "******** Total Running Time of All Kernels = %.5f sec ********\n",
        total_time/1000.0);

    print_matches<<<1, 1>>>();
    cudaDeviceSynchronize();

    cudaFree(X);
    cudaFree(Y);
    cudaFree(S);
    cudaFree(S2);
    cudaFree(r_hist);
    cudaFree(r_prefix_sum);
    cudaFree(r_reduced_prefix);
    cudaFree(r_ordered);
    cudaFree(s_hist);
    cudaFree(s_prefix_sum);
    cudaFree(s_reduced_prefix);
    cudaFree(s_ordered);

    return 0;
}


