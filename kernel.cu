#include <iostream>
#include <vector>
#include <curand_kernel.h>
#include <random>
#include <chrono>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <sstream> 
using namespace std;

//the following code is brought to you by chatgpt
// Constants for the data dimensions
const int IMAGE_SIZE = 784; // 28x28 images
const int NUM_CLASSES = 2;  // Labels are either 0 or 1

// Error-checking macro for CUDA calls
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "  \
                      << cudaGetErrorString(err) << endl;                    \
            exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// Load CSV data into host vectors
bool loadCSV(const string& filename, vector<float>& data, int cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return false;
    }

    string line;
    while (getline(file, line)) {
        istringstream linestream(line);
        string cell;
        int col = 0;
        while (getline(linestream, cell, ',')) {
            data.push_back(stof(cell));  // Convert string to float
            col++;
            if (col >= cols) break;           // Only read up to `cols` columns per row
        }
    }
    file.close();
    return true;
}

// Allocate GPU memory and transfer data to GPU
void transferToGPU(const vector<float>& host_data, float** device_data, int num_elements) {
    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(device_data, num_elements * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(*device_data, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));
}
//and now back to your regular shitty human written code


bool writeCSV(vector<vector<float>> mat)
{
    ofstream file;
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
    ofstream file;
    file.open("fires.csv", ios_base::app);
    for (auto& row : mat) {
        for (auto col : row)
            file << col << ',';
        file << '\n';
    }
    file.close();

    return true;
}


__global__ void synapse_current(float* volt, bool* exin_array, float* fire_con_excite, float* fire_con_inhib, int* fired, float* syn_weights, int num_neurons, int input_size)
{
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_syn_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (post_syn_idx >= num_neurons || pre_syn_idx >= num_neurons + input_size) return;

    int tot_idx = pre_syn_idx + post_syn_idx * (input_size + num_neurons);

    if (fired[pre_syn_idx] == 1)
    {
        if (exin_array[pre_syn_idx] == 1) atomicAdd(&fire_con_excite[post_syn_idx], syn_weights[tot_idx]);


        else atomicAdd(&fire_con_inhib[post_syn_idx], syn_weights[tot_idx]);
    }
}

__global__ void update_neuron(float* volt, float* fire_con_excite, float* fire_con_inhib, float* noise_con, int* fired, float* capac, float* leak, int num_neurons, int V_r, int V_thresh, int V_reset, float del_t, int V_E, int V_I, int offset, int input_size)
{
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_idx >= num_neurons) return;

    fired[neuron_idx + input_size] = 0;
    float leak_con = -1 * leak[neuron_idx] * (volt[neuron_idx] - V_r);
    float fire_con = -1 * (fire_con_excite[neuron_idx] * (volt[neuron_idx] - V_E) + fire_con_inhib[neuron_idx] * (volt[neuron_idx] - V_I));
    float del_v = ((leak_con + fire_con) * del_t / capac[neuron_idx]) + noise_con[offset * num_neurons + neuron_idx];
    volt[neuron_idx] += del_v;
    fire_con_excite[neuron_idx] = 0;
    fire_con_inhib[neuron_idx] = 0;
    if (volt[neuron_idx] > V_thresh)
    {
        volt[neuron_idx] = V_reset;
        fired[neuron_idx + input_size] = 1;
    }
}

__global__ void init_neurons(float* volt, float* fire_con_excite, float* fire_con_inhib, int* fired, int num_neurons, int input_size)
{
    int not_neur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int neur_idx = not_neur_idx - input_size;
    if (not_neur_idx >= num_neurons + input_size) return;
    if (neur_idx >= 0)
    {
        volt[neur_idx] = -70;
        fire_con_excite[neur_idx] = 0;
        fire_con_inhib[neur_idx] = 0;
    }
    fired[not_neur_idx] = 0;
}

__global__ void define_synapses(float* synapses, float temp_ee_weight, float temp_ie_weight, float temp_ei_weight, int temp_num_in_region, int temp_num_regions, int temp_num_inhib, int input_size)
{
    //fix this or maybe the synapse contribution idk, im tired
    int num_neurons = temp_num_inhib + temp_num_in_region * temp_num_regions;
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pre_syn_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (post_syn_idx >= num_neurons || pre_syn_idx >= num_neurons + input_size) return;
    int adjusted_pre_syn_idx = pre_syn_idx - input_size;
    if (adjusted_pre_syn_idx < 0) adjusted_pre_syn_idx = num_neurons + 2;

    //int tot_idx = post_syn_idx * num_neurons + pre_syn_idx;
    int tot_idx = pre_syn_idx + post_syn_idx * (input_size + num_neurons);

    curandState state;
    curand_init(0, post_syn_idx, pre_syn_idx, &state);

    if (post_syn_idx < temp_num_in_region * temp_num_regions)
    {
        if (adjusted_pre_syn_idx == post_syn_idx)
        {
            synapses[tot_idx] = 0; // no self stim
        }

        
        else if (adjusted_pre_syn_idx < temp_num_in_region && post_syn_idx < temp_num_in_region)
        {
            synapses[tot_idx] = temp_ee_weight * curand_uniform(&state); //region 1 self stim
        }
        
        
        
        else if (adjusted_pre_syn_idx >= temp_num_in_region && post_syn_idx >= temp_num_in_region && adjusted_pre_syn_idx < temp_num_in_region * temp_num_regions)
        {
            synapses[tot_idx] = temp_ee_weight * curand_uniform(&state); //region 2 self stim
        }
        
        
        else if (adjusted_pre_syn_idx >= temp_num_in_region * temp_num_regions && adjusted_pre_syn_idx < num_neurons)
        {
            synapses[tot_idx] = temp_ie_weight * curand_uniform(&state); //inhibitory to regions
        }

        else if (adjusted_pre_syn_idx >= num_neurons && post_syn_idx < temp_num_in_region * temp_num_regions)
        {
            synapses[tot_idx] = 0.3 * curand_uniform(&state); //input to regions
        }
        
 
    }

    else
    {
        if (adjusted_pre_syn_idx < temp_num_in_region * temp_num_regions)
        {
            synapses[tot_idx] = temp_ei_weight * curand_uniform(&state); //regions to inhib
        }
    }
}

__global__ void define_static_stuff(float* capac, float* leak, bool* exin_array, int temp_num_in_region, int temp_num_regions, int temp_num_inhib, int input_size)
{
    //here to fix the exin array
    int num_neurons = temp_num_inhib + temp_num_in_region * temp_num_regions;
    int post_syn_idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(0, post_syn_idx, post_syn_idx, &state);
    if (post_syn_idx < num_neurons)
    {

        if (post_syn_idx < temp_num_in_region * temp_num_regions)
        {
            exin_array[input_size + post_syn_idx] = 1;
            capac[post_syn_idx] = 25;
            leak[post_syn_idx] = 0.5;
        }

        else
        {
            exin_array[input_size + post_syn_idx] = 0;
            capac[post_syn_idx] = 20;
            leak[post_syn_idx] = 0.2;
        }
    }
    else if (post_syn_idx < num_neurons + input_size)
    {
        if (curand_uniform(&state) > 0.5) exin_array[post_syn_idx - num_neurons] = 1; //post_syn_idx is a little meaningless here but whatever
        else exin_array[post_syn_idx - num_neurons] = 0; 
        
    }

    else return;
}

__global__ void make_some_noise(float* noise, int num_neurons, int sim_steps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= num_neurons || i >= sim_steps) return;
    int tot_idx = i * num_neurons + j;

    curandState state;
    curand_init(0, i, j, &state);

    noise[tot_idx] = curand_normal(&state) * 0.5 + 0.025; // *std + mean

}

__global__ void make_some_input(int* input, float* input_imgs, int input_size, int sim_steps, int img_idx)
{
    int fake_neur_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int time = blockIdx.y * blockDim.y + threadIdx.y;

    if (fake_neur_idx >= input_size || time >= sim_steps) return;
    int tot_idx = time * input_size + fake_neur_idx;

    curandState state;
    curand_init(0, time, fake_neur_idx, &state);

    if (curand_uniform(&state) < 0.2 * (1 - input_imgs[img_idx * input_size + fake_neur_idx])) //if the random number is greater than 80% of the pixel brightness at that point of the image, the input neuron fires
    {
        input[tot_idx] = 1;
    }
    else input[tot_idx] = 0;

}

__global__ void input_fires(int* fired, int* inputs, int input_size, int time)
{
    //fix fake_neur_idx or something you dumb bastard fuck you i hate you so fucking much
    int fake_neur_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (fake_neur_idx >= input_size) return;

    int tot_idx = time * input_size + fake_neur_idx;
    fired[fake_neur_idx] = inputs[tot_idx];
}

int main() {

    string data_file = "D:\\Neuro Sci\\bi_stable_competition\\mnist_test_ones_zeros_data.csv";
    string label_file = "D:\\Neuro Sci\\bi_stable_competition\\mnist_test_ones_zeros_labels.csv";
    
    vector<float> host_data;
    vector<float> host_labels;

    if (!loadCSV(data_file, host_data, IMAGE_SIZE) || !loadCSV(label_file, host_labels, 1)) {
        return EXIT_FAILURE;
    }

    int num_images = host_labels.size();
    cout << "Loaded " << num_images << " images." << endl;

    float* device_data = nullptr;
    float* device_labels = nullptr;

    // Transfer data and labels to GPU
    transferToGPU(host_data, &device_data, num_images * IMAGE_SIZE);
    transferToGPU(host_labels, &device_labels, num_images);


    unsigned long long seed = 1234;  // Random seed
    int input_size = 784;
    int num_in_region = 1100;
    int num_regions = 2;
    int num_excite = num_regions * num_in_region;
    int num_inhib = 300;
    int num_neurons = num_inhib + num_excite;
    float del_t = 1.0 / 1000; // time step in secons, must be given as a decimal, only a fool would write 1/1000 and be given and integer
    float sim_length = 1.0; // length of simulation in seconds
    int sim_steps = sim_length / del_t;

    cout << sim_steps << endl;
    float ee_weight = 4.8;
    float ie_weight = 5.5;
    float ei_weight = 2.4;

    int V_r = -70;
    int V_reset = -55;
    int V_thresh = -50;
    int V_I = -70;
    int V_E = 0;

    // Host vector to store the result
    vector<vector<float>> all_volts(sim_steps, vector<float>(num_neurons));
    vector<vector<int>> all_fires(sim_steps, vector<int>(num_neurons + input_size));
    vector<float> debug_noise(sim_steps * num_neurons);
    vector<int> debug_input(sim_steps * input_size);

    vector<float> h_volt(num_neurons);
    vector<float> h_fire_con(num_neurons);
    vector<int> h_fired(input_size + num_neurons);

    vector<float> h_capac(num_neurons);
    vector<float> h_leak(num_neurons);

    vector<float> h_synapses((num_neurons) * (input_size + num_neurons));
    vector<bool> h_exin_array(num_neurons);

    int threadsX = 16;
    int threadsY = 16;
    dim3 threads(threadsX, threadsY);

    int blocksX = (input_size + num_neurons + threadsX - 1) / threadsX;
    int blocksY = (input_size + num_neurons + threadsY - 1) / threadsY;
    dim3 blocks(blocksX, blocksY);

    // Device vector
    float* d_volt;
    float* d_fire_con_excite;
    float* d_fire_con_inhib;
    int* d_fired;

    float* d_capac;
    float* d_leak;

    float* d_synapses;
    bool* d_exin_array;

    float* d_pre_comp_noise;
    int* d_pre_comp_input;

    cudaMalloc((void**)&d_volt, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire_con_excite, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fire_con_inhib, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_fired, (input_size + num_neurons) * sizeof(int));
    cudaMalloc((void**)&d_capac, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_leak, num_neurons * sizeof(float));
    cudaMalloc((void**)&d_synapses, (num_neurons) * (input_size + num_neurons) * sizeof(float));
    cudaMalloc((void**)&d_exin_array, (input_size + num_neurons) * sizeof(bool));
    cudaMalloc((void**)&d_pre_comp_noise, num_neurons * sim_steps * sizeof(float));
    cudaMalloc((void**)&d_pre_comp_input, input_size * sim_steps * sizeof(int));

    //initializing data on gpu
    make_some_noise << < blocks, threads >> > (d_pre_comp_noise, num_neurons, sim_steps);
    define_static_stuff << <blocksX, threadsX >> > (d_capac, d_leak, d_exin_array, num_in_region, num_regions, num_inhib, input_size);
    define_synapses << < blocks, threads >> > (d_synapses, ee_weight, ie_weight, ei_weight, num_in_region, num_regions, num_inhib, input_size);
    init_neurons << <blocks, threads >> > (d_volt, d_fire_con_excite, d_fire_con_inhib, d_fired, num_neurons, input_size);
    make_some_input << < blocks, threads >> > (d_pre_comp_input, device_data, input_size, sim_steps, 0);
    cudaMemcpy(debug_noise.data(), d_pre_comp_noise, sim_steps * num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    
    /*
    cudaMemcpy(debug_input.data(), d_pre_comp_input, input_size * sim_steps * sizeof(float), cudaMemcpyDeviceToHost);
    for (int t = 0; t < sim_steps; t++)
    {
        
        cout << debug_input[t] << endl;
    }
    
    
    for (int i = 0; i < 500; i++)
    {
        cout << debug_noise[i] << endl;
    }
    */

    cout << "started" << endl;
    auto start = chrono::high_resolution_clock::now();
    
    for (int t = 0; t < sim_steps; t++)
    {
        //hey it looks really dumb to pass stuff like the number of neurons each time but aparently that saves time over making global variables, who knew!
        input_fires << <blocksX, threadsX >> > (d_fired, d_pre_comp_input, input_size, t);
        update_neuron << <blocksX, threadsX >> > (d_volt, d_fire_con_excite, d_fire_con_inhib, d_pre_comp_noise, d_fired, d_capac, d_leak, num_neurons, V_r, V_thresh, V_reset, del_t, V_E, V_I, t, input_size);
        synapse_current << <blocks, threads >> > (d_volt, d_exin_array, d_fire_con_excite, d_fire_con_inhib, d_fired, d_synapses, num_neurons, input_size);
        //Might need to fix the indexing on the synaptic or the fires, if not just get working on the mnist input

        cudaMemcpy(h_volt.data(), d_volt, num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
        //all_fires[t] = h_fire;
        all_volts[t] = h_volt;

        cudaMemcpy(h_fired.data(), d_fired, (num_neurons + input_size) * sizeof(int), cudaMemcpyDeviceToHost);
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
    //cudaFree(device_data);
    //cudaFree(device_labels);

    return 0;
}