#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_NONCE 256
#define THREADS_PER_BLOCK_MERKLE 256

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

__global__ void merkle_kernel(
    const BYTE* transactions_device,
    int transaction_size_bytes,
    int num_transactions,
    BYTE* merkel_root)
{
    /********************************** INITIAL TRANSACTION HASHING ***********************************/

    extern __shared__ BYTE hash_pool[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Offset in bytes for each thread in the shared memory 
    BYTE* local_hash = hash_pool + threadIdx.x * SHA256_HASH_SIZE;

    if (tid < num_transactions) {
        // Hash the transaction
        const BYTE* transaction = transactions_device + tid * transaction_size_bytes;
        apply_sha256(transaction, local_hash);
    }

    __syncthreads();

    /***************************************** REDUCE PHASE *******************************************/

    // Every level halves the number of hashes
    int num_hashes = num_transactions;

    while (num_hashes > 1) {
        // Each thread processes two hashes
        int in_idx1 = tid * 2;
        int in_idx2 = in_idx1 + 1;
        int next_level_hashes = (num_hashes + 1) / 2;
        
        if (tid < next_level_hashes) {
            BYTE* hash1 = hash_pool + in_idx1 * SHA256_HASH_SIZE;
            // If we have an odd number of hashes, the last thread will hash itself
            BYTE* hash2 = (in_idx2 < num_hashes) ? (hash_pool + in_idx2 * SHA256_HASH_SIZE) : hash1;

            BYTE combined[2 * SHA256_HASH_SIZE];

            #pragma unroll
            for (int i = 0; i < SHA256_HASH_SIZE; ++i)
                combined[i] = hash1[i];
            #pragma unroll
            for (int i = 0; i < SHA256_HASH_SIZE; ++i)
                combined[SHA256_HASH_SIZE + i] = hash2[i];

            BYTE* out_hash = hash_pool + tid * SHA256_HASH_SIZE;
            apply_sha256(combined, out_hash);
        }

        __syncthreads();
        num_hashes = (num_hashes + 1) / 2;
    }

    // Thread 0 writes the Merkle root to the output
    if (threadIdx.x == 0) {
        BYTE* root = hash_pool;
        #pragma unroll
        for (int i = 0; i < SHA256_HASH_SIZE; ++i)
            merkel_root[i] = root[i];
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

    BYTE *transactions_dev, *merkle_root_dev;

    cudaMalloc(&transactions_dev, transactions_count * transaction_size_bytes);
    cudaMemcpy(transactions_dev, transactions_host, transactions_count * transaction_size_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&merkle_root_dev, SHA256_HASH_SIZE);

    int threads_per_block = THREADS_PER_BLOCK_MERKLE;
    int blocks = (transactions_count + threads_per_block - 1) / threads_per_block;

    size_t shared_mem_size = threads_per_block * SHA256_HASH_SIZE;

    merkle_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
        transactions_dev,
        transaction_size_bytes,
        transactions_count,
        merkle_root_dev);

    cudaMemcpy(merkle_root_host, merkle_root_dev, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(transactions_dev);
    cudaFree(merkle_root_dev);
}

// TODO 2: Implement this function in CUDA
int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    char nonce_string[NONCE_SIZE];
    /* This is the serialized CPU version, TODO the gpu version*/
    for (uint32_t nonce = 0; nonce <= max_nonce; nonce++) {
        sprintf(nonce_string, "%u", nonce);
        strcpy((char *)block_content + current_length, nonce_string);
        apply_sha256(block_content, block_hash);

        if (compare_hashes(block_hash, difficulty) <= 0) {
            *valid_nonce = nonce;
            return 0;
        }
    }

    return 1;
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
