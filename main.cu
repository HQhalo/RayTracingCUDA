#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include "./shape/hitable.h"
#include "./shape/hitable_list.h"
#include "./common/vec3.h"
#include "./camera/camera.h"

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

__device__ vec3 color(ray &r){
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void createWorldKernel(hitable &d_world, hitable *d_list,int n){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        d_list[0] = new sphere(vec3(0,0,-1), 0.5);
    }
}

__global__ void renderKernel(vec3 *fbuffer, camera cam, int nx, int ny){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < nx && j < ny){
        float u = 1.0 * i / nx;
        float v = 1.0 * j / ny;

        int idx = j * nx + i;
        ray r = cam.get_ray(u,v);
        fbuffer[idx] = color(r);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   //Each thread gets same seed, a different sequence number, no offset
   curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

int main(){
    int nx = 200;
    int ny = 100;

    vec3 *fbuffer;
    CHECK(cudaMallocManaged(&fbuffer, nx * ny * sizeof(vec3)));
    vec3 lookfrom(13,2,3);
    vec3 lookat(0,0,0);
    vec3 vup(0,1,0);
    camera cam(lookfrom, lookat, vup, 20, float(nx)/float(ny));

    hitable d_world;
    hitable *d_list;
    checkCudaErrors(cudaMalloc((d_list, 2*sizeof(hitable *)));
    checkCudaErrors(cudaMalloc((d_world, sizeof(hitable)));

    createWorldKernel<<<1,1>>>(d_world,d_list,n)

    dim3 blockSize(8,8);
    dim3 gridSize((nx -1) /blockSize.x + 1, (ny - 1)/ blockSize.y + 1);

    renderKernel<<<gridSize,blockSize>>>(fbuffer,cam,nx,ny);
    cudaDeviceSynchronize();

    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col = fbuffer[j * nx + i];
            int ir = 255.99 * col.r();
            int ig = 255.99 * col.g();
            int ib = 255.99 * col.b();
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
    cudaFree(fbuffer);

    return 0;
}
