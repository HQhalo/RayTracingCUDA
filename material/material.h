#ifndef MATERIALH
#define MATERIALH

#include "../common/ray.h"
#include "../shape/hitable.h"
#include <curand_kernel.h>


struct hit_record;
class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *random_state) const = 0;
};

__device__ vec3 random_in_unit_sphere(curandState *random_state) {
    vec3 p;
    do {
        p = 2.0* vec3(curand_uniform(random_state),curand_uniform(random_state),curand_uniform(random_state)) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    return p;
}

__host__ __device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}
__host__ __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__host__ __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}


#endif
