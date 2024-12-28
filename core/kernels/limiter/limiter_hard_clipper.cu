#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" __global__
void limiter_kernel(
    const double* __restrict__ wav_in, // Input audio signal [L0, R0, L1, R1, ..., LN-1, RN-1]
    double* __restrict__ wav_out,      // Output audio signal
    const int n_channels,       // Number of channels (e.g., 2 for stereo)
    const int n_samples,        // Number of samples per channel
    const double threshold,     // Threshold in linear scale (e.g., 0.5 for 50%)
    const double slope,         // Slope parameter for gain calculation
    const double sr,            // Sample rate in Hz
    const double tatt,          // Attack time in ms
    const double trel           // Release time in ms
)
{
    int ch = blockIdx.x;  // Each block processes one channel
    int thread_id = threadIdx.x; // Thread within the block
    int stride = blockDim.x;     // Number of threads in the block

    if (ch >= n_channels) return;

    int start_idx = thread_id * (n_samples / stride);
    int end_idx = (thread_id + 1) * (n_samples / stride);

    double clip_limit = threshold;
    double clip_limit_inv = 1.0 / clip_limit;

    for (int i = start_idx; i < end_idx; ++i)
    {
        int sample_idx = i * n_channels + ch;
        double y = wav_in[sample_idx];

        double abs = fabs(y);
        if(abs >= clip_limit){
            if (y < 0){
                y = -clip_limit;
            }else{
                y = clip_limit;
            }
        }
        wav_out[sample_idx] = y;
    }
}