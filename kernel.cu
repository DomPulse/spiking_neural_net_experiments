#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <fstream>
using namespace std;


bool writeCSV(vector<vector<float>> mat)
{
    std::ofstream file;
    file.open("volts.csv", ios_base::app);
    for (auto& row : mat) {
        for (auto col : row)
            file << col << ',';
        file << '\n';
    }
    file.close();

    return true;
}

bool writeCSVint(vector<vector<int>> mat)
{
    std::ofstream file;
    file.open("fires.csv", ios_base::app);
    for (auto& row : mat) {
        for (auto col : row)
            file << col << ',';
        file << '\n';
    }
    file.close();

    return true;
}


__global__ void matrix_mul_syn_curr(signed int* curr, const bool* fire, const signed char* syn_weights, int* post_syn_idx, int num_neurons, int num_syns)
{
    // Compute each thread's global row and column index
    int syn_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons && syn_idx < num_syns && fire[neuron_idx])
    {
        int tot_idx = neuron_idx * num_syns + syn_idx; //remember this so i can actuall reference shit later
        atomicAdd(&curr[post_syn_idx[tot_idx]], syn_weights[tot_idx]);
    }
}

__global__ void synapse_current(signed int* curr, const bool* fire, const signed char* syn_weights, int* post_syn_idx, int num_neurons, int num_syns)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx < num_neurons)
    {
        if (fire[neuron_idx] == 1)
        {
            for (int syn_idx = 0; syn_idx < num_syns; syn_idx++)
            {
                int tot_idx = neuron_idx * num_syns + syn_idx; //remember this so i can actuall reference shit later
                atomicAdd(&curr[post_syn_idx[tot_idx]], syn_weights[tot_idx]);
            }
        }
    }
}

__global__ void update_neuron(float* volt, float* fire_con, float* noise_con, int* fired, float* capac, float* leak, int num_neurons, int V_r, int V_thresh, int V_reset, float del_t, int offset)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    if (neuron_idx < num_neurons)
    {
        fired[neuron_idx] = 0;
        float leak_con = -1 * leak[neuron_idx] * (volt[neuron_idx] - V_r);
        //float noise_con = curand_normal(&state) * 1 + 0.05; // *std + mean
        float del_v = ((leak_con + fire_con[neuron_idx]) * del_t / capac[neuron_idx]) + noise_con[offset * num_neurons + neuron_idx];
        volt[neuron_idx] += del_v;
        if (volt[neuron_idx] > V_thresh)
        {
            volt[neuron_idx] = V_reset;
            fired[neuron_idx] = 1;
        }
    }
}

__global__ void init_neurons(float* volt, float* fire_con, int* fired, int num_neurons)
{
    int neur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neur_idx >= num_neurons) return;
    volt[neur_idx] = -70;
    fire_con[neur_idx] = 0;
    fired[neur_idx] = 0;
}

__global__ void define_synapses(float* synapses, float temp_ee_weight, float temp_ie_weight, float temp_ei_weight, int temp_num_in_region, int temp_num_regions, int temp_num_inhib)
{
    int num_neurons = temp_num_inhib + temp_num_in_region * temp_num_regions;
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_syn_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (post_syn_idx >= num_neurons || pre_syn_idx >= num_neurons) return;

    int tot_idx = post_syn_idx * num_neurons + pre_syn_idx;

    curandState state;
    curand_init(0, post_syn_idx, pre_syn_idx, &state);

    if (post_syn_idx < temp_num_in_region * temp_num_regions)
    {
        if (pre_syn_idx == post_syn_idx)
        {
            synapses[tot_idx] = 0; // no self stim
        }

        else if ((pre_syn_idx < temp_num_in_region && post_syn_idx < temp_num_in_region) || (pre_syn_idx < temp_num_in_region * temp_num_regions && pre_syn_idx > temp_num_in_region && post_syn_idx > temp_num_in_region))
        {
            synapses[tot_idx] = temp_ee_weight * curand_uniform(&state);
        }

        else if (pre_syn_idx > temp_num_in_region * temp_num_regions)
        {
            synapses[tot_idx] = temp_ie_weight * curand_uniform(&state);
        }
    }

    else
    {
        if (pre_syn_idx < temp_num_in_region * temp_num_regions)
        {
            synapses[tot_idx] = temp_ei_weight * curand_uniform(&state);
        }
    }
    
}

__global__ void define_static_stuff(float* capac, float* leak, bool* exin_array, float temp_ee_weight, float temp_ie_weight, float temp_ei_weight, int temp_num_in_region, int temp_num_regions, int temp_num_inhib)
{
    int num_neurons = temp_num_inhib + temp_num_in_region * temp_num_regions;
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (post_syn_idx >= num_neurons) return;

    if (post_syn_idx < temp_num_in_region * temp_num_regions)
    {
        exin_array[post_syn_idx] = 1;
        capac[post_syn_idx] = 25;
        leak[post_syn_idx] = 0.5;
    }

    else
    {
        exin_array[post_syn_idx] = 0;
        capac[post_syn_idx] = 20;
        leak[post_syn_idx] = 0.2;
    }
}

__global__ void make_some_noise(float* noise, int num_neurons, int sim_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= num_neurons || i >= sim_steps) return;
    int tot_idx = i * num_neurons + j;

    curandState state;
    curand_init(0, i, j, &state);

    noise[tot_idx] = curand_normal(&state) + 0.05; // *std + mean

}

int main() {
    unsigned long long seed = 1234;  // Random seed
    int num_in_region = 1100;
    int num_regions = 2;
    int num_excite = num_regions * num_in_region;
    int num_inhib = 300;
    int num_neurons = num_inhib + num_excite;
    float del_t = 1.0 / 1000; // time step in secons, must be given as a decimal, only a fool would write 1/1000 and be given and integer
    float sim_length = 1.0; // length of simulation in seconds
    int sim_steps = sim_length / del_t;

    cout << sim_steps << endl;
    float ee_weight = 5.3;
    float ie_weight = 5.5;
    float ei_weight = 2.4;

    int V_r = -70;
    int V_reset = -55;
    int V_thresh = -50;

    // Host vector to store the result
    vector<vector<float>> all_volts(sim_steps, vector<float>(num_neurons));
    vector<vector<int>> all_fires(sim_steps, vector<int>(num_neurons));
    vector<float> debug_noise(sim_steps*num_neurons);

    vector<float> h_volt(num_neurons);
    vector<float> h_fire_con(num_neurons);
    vector<int> h_fired(num_neurons);

    vector<float> h_capac(num_neurons);
    vector<float> h_leak(num_neurons);

    vector<float> h_synapses(num_neurons*num_neurons);
    vector<bool> h_exin_array(num_neurons);
      
    int threadsX = 16;
    int threadsY = 16;
    dim3 threads(threadsX, threadsY);

    int blocksX = (num_neurons + threadsX - 1) / threadsX;
    int blocksY = (num_neurons + threadsY - 1) / threadsY;
    dim3 blocks(blocksX, blocksY);

    // Device vector
    float* d_volt;
    float* d_fire_con;
    int* d_fired;

    float* d_capac;
    float* d_leak;

    float* d_synapses;
    bool* d_exin_array;

    float* d_pre_comp_noise;

    cudaMalloc((void**)&d_volt, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire_con, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fired, num_neurons * sizeof(int));
    cudaMalloc((void**)&d_capac, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_leak, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_synapses, num_neurons * num_neurons * sizeof(float));
    cudaMalloc((void**)&d_exin_array, num_neurons * sizeof(bool));
    cudaMalloc((void**)&d_pre_comp_noise, num_neurons * sim_steps * sizeof(float));
    
    

    //initializing data on gpu
    define_static_stuff << <blocks, threads >> > (d_capac, d_leak, d_exin_array, ee_weight, ie_weight, ei_weight, num_in_region, num_regions, num_inhib);
    define_synapses << < blocks, threads >> > (d_synapses, ee_weight, ie_weight, ei_weight, num_in_region, num_regions, num_inhib);
    init_neurons << <blocks, threads >> > (d_volt, d_fire_con, d_fired, num_neurons);
    make_some_noise << < blocks, threads >> > (d_pre_comp_noise, num_neurons, sim_steps);
    cudaMemcpy(debug_noise.data(), d_pre_comp_noise, sim_steps * num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    for (int i = 0; i < 5000; i++) 
    {
        cout << debug_noise[i] << endl;
    }
    */
    
    cout << "started" << endl;
    auto start = chrono::high_resolution_clock::now();
    for (int t = 0; t < sim_steps; t++)
    {
        update_neuron << <blocksX, threadsX >> > (d_volt, d_fire_con, d_pre_comp_noise, d_fired, d_capac, d_leak, num_neurons, V_r, V_thresh, V_reset, del_t, t);

        cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
        //all_fires[t] = h_fire;
        all_volts[t] = h_volt;

        cudaMemcpy(h_fired.data(), d_fired, num_neurons * sizeof(int), cudaMemcpyDeviceToHost);
        //all_fires[t] = h_fire;
        all_fires[t] = h_fired;
    }


    // Copy the result back to the host
    cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cout << h_volt[0] << endl;
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::seconds>(stop - start);
    cout << duration.count() << endl;
    cout << "done" << endl;
    writeCSV(all_volts);
    writeCSVint(all_fires);
    

    // Cleanup
    cudaFree(d_volt);
    cudaFree(d_fired);
    cudaFree(d_capac);
    cudaFree(d_leak);
    cudaFree(d_synapses);
    cudaFree(d_exin_array);

    return 0;
}