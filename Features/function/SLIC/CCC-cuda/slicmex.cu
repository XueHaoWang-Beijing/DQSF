__global__ void addVectorsMask(float *A, float *B, float *C, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i!= size)
        return;

    C[i] = A[i] + B[i];
}

void addVectors(float *A, float *B, float *C, int size)
{
    float *devPtrA = 0,*devPtrB = 0,*devPtrC = 0;
    cudaMalloc(&devPtrA,sizeof(float)* size);
    cudaMalloc(&devPtrB,sizeof(float)* size);
    cudaMalloc(&devPtrC,sizeof(float)* size);

    cudaMemcpy(devPtrA,A, sizeof(float)* size, cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB,B, sizeof(float)* size, cudaMemcpyHostToDevice);
    addVectorsMask<<<int(size/1024),1024>>>(devPtrA,devPtrB, devPtrC, size);
    cudaMemcpy(C,devPtrC, sizeof(float)* size, cudaMemcpyDeviceToHost);

    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);

}