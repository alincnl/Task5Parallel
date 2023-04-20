/* 
 * Реализация уравнения теплопроводности в двумерной области
 * на равномерных сетках с использованием CUDA. 
 * Операция редукции (вычисление максимального значения ошибки)
 * для одного MPI процесса реализуется с помощью библиотеки CUB.
 * Подсчет глобального значения ошибки, обмен граничными условиями
 * реализуется с использованием NCCL 
*/

// подключение библиотек
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include <nccl.h>

using namespace std;

// функция, обновляющая значения сетки
__global__ void update(double* A, double* Anew, int start, int stop, int size) 
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i*size + j >= start && i*size + j < stop)
        if (j < size - 1 && j > 0 && i > 0 && i < size - 1){
            double left = A[i * size + j - 1];
            double right = A[i * size + j + 1];
            double top = A[(i - 1) * size + j];
            double bottom = A[(i + 1) * size + j];
            Anew[i*size + j] = 0.25 * (left + right + top + bottom);
        }
}


// функция нахождения разности двух массивов
__global__ void substract(double* A, double* Anew, double* res, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= 0 && i < size && j >= 0 && j < size)
		res[i*size + j] = Anew[i*size + j] - A[i*size + j];
}

__constant__ double add;

// функция для заполнения массивов
__global__ void fill(double* A, double* Anew, int size) 
{
    A[0] = 10;
    A[size - 1] = 20;
    A[(size - 1)*(size)] = 20;
    A[(size - 1)*(size)+ size - 1] = 30;

    Anew[0] = 10;
    Anew[size - 1] = 20;
    Anew[(size - 1)*(size)] = 20;
    Anew[(size - 1)*(size)+ size - 1] = 30;

	for (size_t i = 1; i < size - 1; i++) {
		A[i] = A[i - 1] + add;
        A[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        A[i*(size)] = A[(i - 1) *(size)] + add;
        A[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
        Anew[i] = A[i - 1] + add;
        Anew[(size - 1)*(size)+i] = A[(size - 1)*(size)+i - 1] + add;
        Anew[i*(size)] = A[(i - 1) *(size)] + add;
        Anew[i*(size)+size - 1] = A[(i - 1)*(size)+size - 1] + add;
	}
}

// основное тело программы
int main(int argc, char *argv[]){
    // время до выполнения программы
    auto begin = std::chrono::steady_clock::now();

    // инициализация MPI
    int myRank, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Status status;

    if(nRanks < 1 || nRanks > 4) {
        printf("1-4");
        exit(0);
    }

    ncclUniqueId id;
    if (myRank == 0) 
        ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    cudaSetDevice(myRank);

    // инициализация переменных
    double tol = atof(argv[1]);
    const int size = atoi(argv[2]), iter_max = atoi(argv[3]);

    int start, stop, step = size*size / nRanks;
    start = myRank * step;
    stop = (myRank + 1) * step;

    double *d_A = NULL, *d_Anew = NULL, *d_Asub;
    cudaError_t cudaerr = cudaSuccess;

    // выделение памяти под массивы и проверка на наличие ошибок
    cudaerr = cudaMalloc((void **)&d_A, sizeof(double)*(size+2)*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    cudaerr = cudaMalloc((void **)&d_Anew, sizeof(double)*(size+2)*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    cudaerr = cudaMalloc((void **)&d_Asub, sizeof(double)*size*size);
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
                cudaGetErrorString(cudaerr));
        exit(EXIT_FAILURE);
    }

    d_Anew = d_Anew + size;
    d_A = d_A + size;
    // инициализация переменных
    int iter = 0;
    double error = 1;
    double addH = 10.0 / (size - 1);
    cudaMemcpyToSymbol(add, &addH, sizeof(double));

    dim3 threadPerBlock = dim3(32, 32);
    dim3 blocksPerGrid = dim3((size+threadPerBlock.x-1)/threadPerBlock.x,(size+threadPerBlock.y-1)/threadPerBlock.y);
    
    fill<<< 1, 1 >>> (d_Anew, d_A, size);

    double* d_error;
    cudaMalloc(&d_error, sizeof(double));

    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, id, myRank);

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Asub, d_error, size*size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // цикл пересчета ошибки и обновления сетки 
    while((error > tol) && (iter < iter_max)) {
        iter = iter + 1;
        
        // обновление сетки и передача граничных значений
        update<<< threadPerBlock, blocksPerGrid >>> (d_Anew, d_A, start, stop, size);

        ncclGroupStart();
        // нижняя граница
        if(myRank != nRanks-1)
            ncclSend(d_Anew + stop - size, size, ncclDouble, (myRank+1)%nRanks, comm, 0);
        if(myRank != 0)
            ncclRecv(d_Anew + start - size, size, ncclDouble, (myRank-1 + nRanks)%nRanks, comm, 0);

        // верхняя граница
        if(myRank != 0)
            ncclSend(d_Anew + start, size, ncclDouble, (myRank-1)%nRanks, comm, 0);
        if(myRank != nRanks-1)
            ncclRecv(d_Anew + stop, size, ncclDouble, (myRank+1 + nRanks)%nRanks, comm, 0);
        ncclGroupEnd();
        
        // пересчет значения ошибки раз в 100 итераций
        if(iter % 100 == 0){
            substract<<< threadPerBlock, blocksPerGrid >>> (d_Anew, d_A, d_Asub, size);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Asub, d_error, size*size);
            ncclAllReduce(d_error, d_error, 1, ncclDouble, ncclMax, comm, 0);
            cudaMemcpyAsync(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
        }
        // обмен значениями
        double* swap = d_A;
        d_A = d_Anew;
        d_Anew = swap;
    }

    // освобождаем память
    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);
    cudaFree(d_Asub);
    cudaFree(d_error);
    cudaFree(d_temp_storage);
    ncclCommDestroy(comm);
    MPI_Finalize();

    // считаем и выводим затраченное время с помощью std::chrono
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    if(myRank==0){
        std::cout << iter << ":" << error << "\n";
        std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    }
    return 0;
}
