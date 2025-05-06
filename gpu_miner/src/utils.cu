#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK_NONCE 256
#define THREADS_PER_BLOCK_MERKLE_INIT 256
#define THREADS_PER_BLOCK_MERKLE_REDUCE 128

// --- Helper Macro for CUDA Error Checking ---
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        cudaDeviceReset(); // Resets the device on error
        exit(99);
    }
}

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
    const BYTE* d_input_hashes,
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

    // Buffers for the two input hashes
    BYTE hash1[SHA256_HASH_SIZE];
    BYTE hash2[SHA256_HASH_SIZE];

    // Load hash1
    if (in_idx1 < num_input_hashes) {
        d_strcpy((char*)hash1, (const char*)&d_input_hashes[in_idx1 * SHA256_HASH_SIZE]);
    } else {
        hash1[0] = '\0'; // Padding
    }

    // Load hash2
    if (in_idx2 < num_input_hashes) {
        d_strcpy((char*)hash2, (const char*)&d_input_hashes[in_idx2 * SHA256_HASH_SIZE]);
    } else {
        // Odd number of hashes, so we need to duplicate hash1
        d_strcpy((char*)hash2, (char*)hash1);
    }

    // Shouldn't be necessary, but just in case
    if (hash1[0] != '\0') {
        BYTE combined[(SHA256_HASH_SIZE - 1) * 2 + 1];
        d_strcpy((char*)combined, (char*)hash1);
        d_strcat((char*)combined, (char*)hash2);
        apply_sha256(combined, &d_output_hashes[out_idx * SHA256_HASH_SIZE]);
    } else {
        d_output_hashes[out_idx * SHA256_HASH_SIZE] = '\0';
    }
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

    checkCudaErrors(cudaMalloc((void**)&transactions_dev, transactions_count * transaction_size_bytes));
    checkCudaErrors(cudaMemcpy(transactions_dev, transactions_host, transactions_count * transaction_size_bytes, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&dev_hashes_ping, transactions_count * SHA256_HASH_SIZE));

    int threads_per_block = THREADS_PER_BLOCK_MERKLE_INIT;
    /* Ceil the number of blocks */
    int blocks_no = (transactions_count + threads_per_block - 1) / threads_per_block;

    initial_hash_kernel<<<blocks_no, threads_per_block>>>(transactions_dev, transaction_size_bytes, transactions_count, dev_hashes_ping);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(transactions_dev));

    int n_hashes = transactions_count;

    /************************************* REDUCTION PHASE ***********************************/

    // Max hashes for next level (output of reduction)
    checkCudaErrors(cudaMalloc((void**)&dev_hashes_pong, ((n_hashes + 1) / 2) * SHA256_HASH_SIZE));

    int threads_per_block_reduce = THREADS_PER_BLOCK_MERKLE_REDUCE;

    while (n_hashes > 1) {
        /* The number of hashes that will result on this level */
        int out_hashes_on_this_level = (n_hashes + 1) / 2;
        /* Ceil the number of blocks */
        int blocks_no_reduce = (out_hashes_on_this_level + threads_per_block_reduce - 1) / threads_per_block_reduce;
        
        if (blocks_no_reduce > 0) {
            reduce_merkle_level_kernel<<<blocks_no_reduce, threads_per_block_reduce>>>(
                dev_hashes_ping,
                n_hashes,
                dev_hashes_pong);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

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
    checkCudaErrors(cudaMemcpy(merkle_root_host, dev_hashes_ping, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_hashes_ping));
    checkCudaErrors(cudaFree(dev_hashes_pong));
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
