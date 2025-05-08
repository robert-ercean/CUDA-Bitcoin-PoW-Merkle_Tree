# Overview

CUDA implementation of two critical phases of the bitcoin mining process:
1.  **Merkle Root Construction**: Generates a Merkle root from a list of transactions.
2.  **Nonce Finding (Proof-of-Work)**: Searches for a nonce that satisfies a given difficulty target for a block header.

## Merkle Root Construction (`construct_merkle_root`)

### Overall Process:

Split into 2 phases, each with their respective kernels:
1.  **Initial Transaction Hashing**: Each transaction is individually hashed.
2.  **Iterative Reduction**: Pairs of hashes from the previous level are concatenated and then hashed together. This process repeats, with each level having ceil(half the number) of hashes as the previous one, until only a single hash remains – `the Merkle root`.

### CUDA Kernels:

1.  **`initial_hash_kernel`**:
    * Each CUDA thread processes one transaction from the input list.
    * The standard `256` threads per block is assigned
    * Then, the number of blocks required for the kernel launch is calculated by dividing the total number of transactions by the number of threads per block. 
    * These initial hashes are stored in a device buffer.

2.  **`reduce_merkle_level_kernel`**:
    * This kernel performs one level of the reduction tree.
    * Each thread is responsible for parsing two input hashes to generate one output hash for the current level.
    * The two input hashes are fetched from the previous level's output using two **Ping-Pong** style buffers that alternate
    between input and output for each reduction level.
        * If there's an odd number of input hashes, the last hash is duplicated to form a pair.
    * The two input  hashes are concatenated and null-terminated
    * The concatenated string of hex strings is then hashed using `apply_sha256` to produce the next level's hash
    * The resulted hash is stored in the output buffer.
### Wrapper (`construct_merkle_root`):
* Manages memory allocation and data transfers between host and device.
* Launches `initial_hash_kernel` once.
* Repeatedly launches `reduce_merkle_level_kernel` in a loop, swapping ping-pong buffers, until `n_hashes` becomes 1.
* Copies the Merkle Root to the solution variable.

## Nonce Finding (Proof-of-Work)

### CUDA Kernel (`find_nonce_kernel`):

* **Parallel Search Strategy**:
    * We have a total sarch space of `max_nonce` values to find a valid nonce.
    * These values are distributed among all threads across all launched blocks.
    * Each thread calculates its initial nonce based on its global thread ID (`blockIdx.x * blockDim.x + threadIdx.x`).
    * Since a thread will process multiple nonces, it will take strides of the total number of threads, offseting its initial value
    by the total number of threads in the grid to avoid overlapping work.
* **Dynamic Shared Memory**:
    * To speed-up memory acces, the portion of the block header that is constant for all possible nonces i.e. the `header template` is
    stored in shared memory per block.
    * After being launched, all of the threads within a block will dynamically load teh block header template into their shared memory
    using the same stride technique mentioned above
* **Synchronization**:
    * A global `found_flag` (an integer in global device memory) is used to signal all threads when a solution is found:
        * At the start of the procedure and at the beginning of each nonce value iteration, a thread checks the `found_flag` to see if a solution has already been found. If it has, the thread exits early. 
        * If a thread finds a nonce that satisfies the difficulty (`compare_hashes <= 0`), it attempts to set the `found_flag` to 1 using atomic operations, thus ensuring that the first nonce found is the one that will be returned, and all other threads will then exit.

### Host Orchestration (`find_nonce` function):

* Wrapper function for the kernel
* Sets up device memory for its parameters
* The count of worker threads is given by the product of the macros `(THREADS_PER_BLOCK_NONCE, TOTAL_BLOCKS_NONCE)`
    * After some fiddling and testing around, I've found that the best performance is achieved with `THREADS_PER_BLOCK_NONCE = 256` and `TOTAL_BLOCKS_NONCE = 64`
* The function returns `0` if a nonce was found, and `1` otherwise.

## Github: https://github.com/robert-ercean/CUDA-Bitcoin-PoW-Merkle_Tree