#include <thrust/sort.h>
#include <thrust/execution_policy.h>

// The one thrust call, in its own file: nvcc compiles the slow thrust headers
// only when this files changes, so edits to main.cu rebuild in secounds

void sortByCell(int* cellId, int* perm, int N){
    thrust::sort_by_key(thrust::device, cellId, cellId + N, perm);
}