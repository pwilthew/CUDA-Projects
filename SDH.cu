/* =======================================================
    Student: Patricia Wilthew
    The basic SDH algorithm implementation for 3D data
    To compile: nvcc SDH.c -o SDH
   =======================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#define BOX_SIZE 23000


/* 
    Structure: atom.
    Descriptors for single atom in the tree.
*/
typedef struct atomdesc
{
    double x_pos;
    double y_pos;
    double z_pos;
} atom;

/*
    Structure: bucket.
    Size of the buckets.
*/
typedef struct hist_entry 
{
    long long d_cnt;
} bucket;


cudaError_t err;
long long PDH_acnt;
double PDH_res;
int num_buckets, PDH_threads;
bucket *histogram;
atom *atom_list;

struct timezone Idunno; 
struct timeval startTime, endTime;


/*
    Method: distance.
    Distance of two points (x1, y1, z1) and (x2, y2, z2).
*/
__device__
double distance(double x1, double y1, double z1, double x2, double y2, double z2)
{
    return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


/*
    Method: PDH_on_gpu.
    SDH solution in GPU threads.
*/
__global__
void PDH_on_gpu(double *x, double *y, double *z, bucket *hist,
                int PDH_acnt, double PDH_res, int num_buckets)
{
    extern __shared__ unsigned int SHMOut[];

    int t_id, b_id, t, s;
    int i, h_pos;
    double x1, y1, z1, x2, y2, z2, d;

    t_id = threadIdx.x;
    b_id = blockIdx.x;
    t = b_id*blockDim.x + t_id;

    // Initialize Shared Memory to Zero.
    for (s = 0; s < (num_buckets + blockDim.x - 1)/blockDim.x; s++)
    {
        if (t_id + s*blockDim.x < num_buckets) 
        {
            SHMOut[t_id + s*blockDim.x] = 0;
        }
    }

    // The t-th datum of b-th input data block.
    i = t + 1;
    x1 = x[t];
    y1 = y[t];
    z1 = z[t];

    for (i=t+1; i < PDH_acnt; i++)
    {
        x2 = x[i];
        y2 = y[i];
        z2 = z[i];

        d = distance(x1, y1, z1, x2, y2, z2);
        h_pos = (int) (d / PDH_res);
        atomicAdd(&SHMOut[h_pos], 1);
    }

    __syncthreads();

    // Write results to Global Memory.
    for (s = 0; s < (num_buckets + blockDim.x - 1)/blockDim.x; s++)
    {
        if (t_id + s*blockDim.x < num_buckets)
        {
            atomicAdd((unsigned int *)&hist[t_id + s*blockDim.x].d_cnt,
                      SHMOut[t_id + s*blockDim.x]);
        }
    }
}


/* 
    Method: p2p_distance.
    Distance of two points in the atom_list.
*/
double p2p_distance(atom *atom_list, int ind1, int ind2)
{    
    double x1 = atom_list[ind1].x_pos;
    double x2 = atom_list[ind2].x_pos;
    double y1 = atom_list[ind1].y_pos;
    double y2 = atom_list[ind2].y_pos;
    double z1 = atom_list[ind1].z_pos;
    double z2 = atom_list[ind2].z_pos;
        
    return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/*
    Method: PDH_baseline.
    Brute-force SDH solution in a single CPU thread.
*/
int PDH_baseline(atom *atom_list, bucket *histogram, long long PDH_acnt, double PDH_res)
{
    int i, j, h_pos;
    double dist;
    
    for (i = 0; i < PDH_acnt; i++)
    {
        for (j = i+1; j < PDH_acnt; j++)
        {
            dist = p2p_distance(atom_list, i,j);
            h_pos = (int) (dist / PDH_res);
            histogram[h_pos].d_cnt++;
        } 
    }
    return 0;
}


/*
    Method: report_running_time.
    Set a checkpoint and show the (natural) running time in seconds.
*/
double report_running_time()
{
    long sec_diff, usec_diff;
    gettimeofday(&endTime, &Idunno);
    sec_diff = endTime.tv_sec - startTime.tv_sec;
    usec_diff= endTime.tv_usec-startTime.tv_usec;
    if (usec_diff < 0)
    {
        sec_diff--;
        usec_diff += 1000000;
    }
    printf("Running time: %ld.%06ld\n", sec_diff, usec_diff);
    return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}

/*
    Method: output_histogram.
    Print the counts in all buckets of the histogram.
*/
void output_histogram(bucket *histogram, int num_buckets)
{
    int i; 
    long long total_cnt = 0;
    for (i=0; i< num_buckets; i++)
    {
        if (i%5 == 0)  // Print 5 buckets in a row.
            printf("\n%02d: ", i);
        printf("%15lld ", histogram[i].d_cnt);
        total_cnt += histogram[i].d_cnt;
        //  Also want to make sure the total distance count is correct.
        if (i == num_buckets - 1)    
            printf("\n T:%lld \n", total_cnt);
        else printf("| ");
    }
}

/*
    Method: catch_error.
    Prints any CUDA error to stdout.
*/
void catch_error(cudaError_t error)
{
    if (error)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
}


int main(int argc, char **argv)
{
    if (argc <= 3)
    {
        printf("Usage: ./SDH {# Atoms} {# Buckets} {# Threads}\n");
        exit(1);
    }

    if (atoi(argv[3]) < 32)
    {
        printf("Number of threads must be greater or equal to 32.\n");
        exit(1);
    }

    PDH_acnt = atoi(argv[1]);
    PDH_res = atof(argv[2]);
    PDH_threads = atoi(argv[3]);

    // Variables declaration;
    float time = 0;
    int i;
    double *x, *y, *z, *d_x, *d_y, *d_z;
    bucket *d_histogram, *h_histogram;
    // bucket *difference_histogram;

    // Variables initialization and mem allocation.
    num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
    atom_list = (atom *)malloc(sizeof(atom) * PDH_acnt);
    histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    x = (double *)malloc(sizeof(double)*PDH_acnt);
    y = (double *)malloc(sizeof(double)*PDH_acnt);
    z = (double *)malloc(sizeof(double)*PDH_acnt);
    h_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    // difference_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
    err = cudaSuccess;

    srand(1);
    // Generate data following a uniform distribution.
    for (i = 0;  i < PDH_acnt; i++)
    {
        x[i] = atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        y[i] = atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
        z[i] = atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
    }

    /*
    printf("----CPU----");

    // Start counting time.
    gettimeofday(&startTime, &Idunno);

    // Call CPU single thread version to compute the histogram.
    PDH_baseline(atom_list, histogram, PDH_acnt, PDH_res);

    // Check the total running time.
    report_running_time();

    // Print out the histogram.
    output_histogram(histogram, num_buckets);
    */

    /* My part of the project */

    // Initialize h_histogram with zeroes.
    for (i = 0; i < num_buckets; i++)
    {
        h_histogram[i].d_cnt = 0;
    }

    // Allocate memory in device for single dim arrays.
    err = cudaMalloc((void **)&d_x, PDH_acnt * sizeof(double)); catch_error(err);
    err = cudaMalloc((void **)&d_y, PDH_acnt * sizeof(double)); catch_error(err);
    err = cudaMalloc((void **)&d_z, PDH_acnt * sizeof(double)); catch_error(err);

    // Allocate memory in device for histogram.
    err = cudaMalloc(&d_histogram, num_buckets * sizeof(bucket)); catch_error(err);
    
    // Copy single dim arrays to device.
    err = cudaMemcpy(d_x, x, PDH_acnt * sizeof(double), cudaMemcpyHostToDevice); catch_error(err);
    err = cudaMemcpy(d_y, y, PDH_acnt * sizeof(double), cudaMemcpyHostToDevice); catch_error(err);
    err = cudaMemcpy(d_z, z, PDH_acnt * sizeof(double), cudaMemcpyHostToDevice); catch_error(err);

    // Copy zeroed histogram from host to device.
    err = cudaMemcpy(d_histogram, h_histogram, num_buckets * sizeof(bucket),
                     cudaMemcpyHostToDevice); catch_error(err);

    // Recording variables.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start to record.
    cudaEventRecord( start, 0);

    // Call GPU version.
    PDH_on_gpu<<<(PDH_acnt - 1 + PDH_threads)/PDH_threads,
                  PDH_threads,
                  num_buckets * sizeof(int)>>>(d_x, d_y, d_z,
                                               d_histogram,
                                               PDH_acnt,
                                               PDH_res,
                                               num_buckets);

    // Stop recording.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy histogram from device to host.
    err = cudaMemcpy(h_histogram, d_histogram, num_buckets * sizeof(bucket),
                     cudaMemcpyDeviceToHost); catch_error(err);

    // Print out the histogram.
    output_histogram(h_histogram, num_buckets);

    // Output the total running time.
    printf("******** Total Running Time of Kernel = %.5f sec *******\n", time/1000.0);

    /*
    printf("\n----Difference between histograms:\n");

    // Print the difference between the histograms.
    for (i = 0; i < num_buckets; i++)
    {
        difference_histogram[i].d_cnt = abs(histogram[i].d_cnt - h_histogram[i].d_cnt);
    }

    // Print out the histograms' difference.
    output_histogram(difference_histogram, num_buckets);   
    */

    // Free memory.
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_histogram);
    free(histogram);
    free(h_histogram);
    free(atom_list);
    free(x);
    free(y);
    free(z);

    return 0;
}

