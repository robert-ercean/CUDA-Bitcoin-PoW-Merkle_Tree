#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_NONCE 256
#define THREADS_PER_BLOCK_MERKLE_INIT 256
#define THREADS_PER_BLOCK_MERKLE_REDUCE 256
#define TOTAL_BLOCKS_NONCE 64

// CUDA sprintf alternative for nonce finding. Converts integer to its string representation. Returns string's length.
__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 2;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    // Reverse the string
    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

// CUDA strlen implementation.
__host__ __device__ size_t d_strlen(const char *str) {
    size_t len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

// CUDA strcpy implementation.
__device__ void d_strcpy(char *dest, const char *src){
    int i = 0;
    while ((dest[i] = src[i]) != '\0') {
        i++;
    }
}

// CUDA strcat implementation.
__device__ void d_strcat(char *dest, const char *src){
    while (*dest != '\0') {
        dest++;
    }
    while (*src != '\0') {
        *dest = *src;
        dest++;
        src++;
    }
    *dest = '\0';
}

// Compute SHA256 and convert to hex
__host__ __device__ void apply_sha256(const BYTE *input, BYTE *output) {
    size_t input_length = d_strlen((const char *)input);
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];  // High nibble
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];         // Low nibble
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0'; // Null-terminate
}

// Compare two hashes
__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; // hash1 is lower
        } else if (hash1[i] > hash2[i]) {
            return 1; // hash2 is lower
        }
    }
    return 0; // hashes are equal
}

__global__ void reduce_merkle_level_kernel(
    BYTE* d_input_hashes,
    int num_input_hashes,
    BYTE* d_output_hashes)
{
    const int tid = threadIdx.x;
    const int out_idx = blockIdx.x * blockDim.x + tid;

    const int in_idx1 = out_idx * 2;
    const int in_idx2 = in_idx1 + 1;
    
    // Each thread processes 2 input hashes to output 1 hash
    const int out_hashes_count = (num_input_hashes + 1) / 2;
    
    // Sanity check
    if (out_idx >= out_hashes_count) return;

    // Pointers to this thread's input hashes
    BYTE *hash1;
    BYTE *hash2;

    hash1 = d_input_hashes + in_idx1 * SHA256_HASH_SIZE;
    if (in_idx2 < num_input_hashes) {
        hash2 = d_input_hashes + in_idx2 * SHA256_HASH_SIZE;
    } else {
        // If there's no second hash, we need to duplicate the first one
        hash2 = hash1;
    }

    // Concatenate the two hashes
    BYTE concatenated_hashes[SHA256_HASH_SIZE * 2 - 1];

    #pragma unroll
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
        concatenated_hashes[i] = hash1[i];
        concatenated_hashes[i + SHA256_HASH_SIZE - 1] = hash2[i];
    }
    concatenated_hashes[(2 * SHA256_HASH_SIZE) - 2] = '\0';

    // Compute the SHA256 hash of the concatenated hashes
    apply_sha256(concatenated_hashes, d_output_hashes + out_idx * SHA256_HASH_SIZE);
}


/* Level 0 hashing of the initial transactions */
__global__ void initial_hash_kernel(const BYTE* transactions_device, int transaction_size_bytes, int num_transactions, BYTE* dev_out_hashes) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < num_transactions) {
        const BYTE* src = transactions_device + tid * transaction_size_bytes;
        BYTE* dest = dev_out_hashes + tid * SHA256_HASH_SIZE;
        
        apply_sha256(src, dest);
    }
}

/*
 * @param transaction_size_bytes Size of each transaction string in bytes including null terminator
 * @param transactions_host Pointer to the host memory containing the transaction strings
 * @param max_transactions_in_a_block Maximum number of transactions in a block
 * @param n The number of actual transactions in the block
 * @param merkle_root_host Pointer to the host memory where the Merkle root will be stored
 * */
void construct_merkle_root(int transaction_size_bytes, BYTE *transactions_host, int max_transactions_in_a_block, int transactions_count, BYTE merkle_root_host[SHA256_HASH_SIZE]) {

    /****************************** INITIAL TRANSACTION HASHING ******************************/
    BYTE *transactions_dev;
    /* Ping Pong style buffers */
    BYTE *dev_hashes_ping, *dev_hashes_pong;

    cudaMalloc((void**)&transactions_dev, transactions_count * transaction_size_bytes);
    cudaMemcpy(transactions_dev, transactions_host, transactions_count * transaction_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&dev_hashes_ping, transactions_count * SHA256_HASH_SIZE);

    int threads_per_block = THREADS_PER_BLOCK_MERKLE_INIT;
    /* Ceil the number of blocks */
    int blocks_no = (transactions_count + threads_per_block - 1) / threads_per_block;

    initial_hash_kernel<<<blocks_no, threads_per_block>>>(transactions_dev, transaction_size_bytes, transactions_count, dev_hashes_ping);
    cudaDeviceSynchronize();
    cudaFree(transactions_dev);

    int n_hashes = transactions_count;

    /************************************* REDUCTION PHASE ***********************************/

    // Max hashes for next level (output of reduction)
    cudaMalloc((void**)&dev_hashes_pong, ((n_hashes + 1) / 2) * SHA256_HASH_SIZE);

    int threads_per_block_reduce = THREADS_PER_BLOCK_MERKLE_REDUCE;

    while (n_hashes > 1) {
        /* The number of hashes that will result on this level */
        int out_hashes_on_this_level = (n_hashes + 1) / 2;
        /* Ceil the number of blocks */
        int blocks_no_reduce = (out_hashes_on_this_level + threads_per_block_reduce - 1) / threads_per_block_reduce;
        
        reduce_merkle_level_kernel<<<blocks_no_reduce, threads_per_block_reduce>>>(
            dev_hashes_ping,
            n_hashes,
            dev_hashes_pong);
        cudaDeviceSynchronize();

        n_hashes = out_hashes_on_this_level;

        // Swap the input and output buffers for the next iter
        BYTE* temp_ptr = dev_hashes_ping;
        dev_hashes_ping = dev_hashes_pong;
        dev_hashes_pong = temp_ptr;

        if (n_hashes == 1) {
            break;
        }
    }
    // In the end, the first hash in dev_hashes_ping is the Merkle root
    cudaMemcpy(merkle_root_host, dev_hashes_ping, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(dev_hashes_ping);
    cudaFree(dev_hashes_pong);
}

__global__ void find_nonce_kernel(
    BYTE* difficulty,
    BYTE* block_content_template,
    size_t current_length,
    uint32_t max_nonce,
    BYTE* out_hash,
    int* found_flag,
    uint32_t* result_nonce)
{
    if (atomicAdd(found_flag, 0)) return;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total_threads = gridDim.x * blockDim.x;

    extern __shared__ BYTE shared_block_template[]; // Dynamic shared memory

    // Distribute the work among this block's threads
    // Each thread copies its part of the block content template to shared memory
    for (int i = threadIdx.x; i < current_length; i += blockDim.x) {
        shared_block_template[i] = block_content_template[i];
    }

    // Threads within this block wait at the barrier before doing any work with the shared memory
    __syncthreads();

    BYTE local_block[BLOCK_SIZE];
    BYTE local_hash[SHA256_HASH_SIZE];
    char nonce_str[NONCE_SIZE];

    for (uint32_t nonce = tid; nonce <= max_nonce; nonce += total_threads) {
        // Exit if we've already found a nonce
        if (atomicAdd(found_flag, 0) == 1) return;

        // Copy from shared memory to local block
        for (size_t i = 0; i < current_length; i++) {
            local_block[i] = shared_block_template[i];
        }

        int nonce_len = intToString(nonce, nonce_str);
        memcpy(local_block + current_length, nonce_str, nonce_len);
        local_block[current_length + nonce_len] = '\0';

        apply_sha256(local_block, local_hash);

        // Atomically check if a nonce has not already been found, and set the flag to 1
        if (compare_hashes(local_hash, difficulty) <= 0 && (atomicExch(found_flag, 1) == 0)) {
            *result_nonce = nonce;
            memcpy(out_hash, local_hash, SHA256_HASH_SIZE);
        }
    }
}



int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    // Initialize device memory pointers and alloc device memory
    BYTE *dev_difficulty, *dev_block_content, *dev_block_hash;
    uint32_t *dev_valid_nonce;
    int *dev_found_flag;
    
    cudaMalloc((void **)&dev_block_content, current_length + NONCE_SIZE);
    cudaMalloc((void **)&dev_difficulty, SHA256_HASH_SIZE);
    cudaMalloc((void **)&dev_valid_nonce, sizeof(uint32_t));
    cudaMalloc((void **)&dev_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void **)&dev_found_flag, sizeof(int));
    
    cudaMemcpy(dev_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_block_content, block_content, current_length, cudaMemcpyHostToDevice);
    
    int host_found_flag = 0;
    cudaMemcpy(dev_found_flag, &host_found_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    const int threads_per_block = THREADS_PER_BLOCK_NONCE;
    const int blocks = TOTAL_BLOCKS_NONCE;

    // The size of the shared array per block for the block content template
    size_t shared_mem_size = current_length * sizeof(BYTE);
    find_nonce_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        dev_difficulty,
        dev_block_content,
        current_length,
        max_nonce,
        dev_block_hash,
        dev_found_flag,
        dev_valid_nonce
    );
    cudaDeviceSynchronize();
    cudaMemcpy(&host_found_flag, dev_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Return the results
    cudaMemcpy(valid_nonce, dev_valid_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_hash, dev_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(dev_difficulty);
    cudaFree(dev_block_content);
    cudaFree(dev_block_hash);
    cudaFree(dev_valid_nonce);
    cudaFree(dev_found_flag);
    
    return host_found_flag ? 0 : 1;
}

__global__ void dummy_kernel() {}

// Warm-up function
void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}