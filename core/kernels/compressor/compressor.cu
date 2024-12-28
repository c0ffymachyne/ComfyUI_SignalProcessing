#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" __global__
void compexp_kernel(
    const double* wav_in,       // Input audio signal [L0, R0, L1, R1, ..., LN-1, RN-1]
    double* wav_out,            // Output audio signal
    const int n_channels,       // Number of channels (e.g., 2 for stereo)
    const int n_samples,        // Number of samples per channel
    const double comp,          // Compression/expansion factor
    const double release,       // Release time in ms
    const double attack,        // Attack time in ms
    const double a,             // Filter parameter < 1
    const double Fs             // Sampling rate in Hz
) {
    int ch = blockIdx.x;  // Each block processes one channel
    int thread_id = threadIdx.x; // Thread within the block
    int stride = blockDim.x;     // Number of threads in the block

    if (ch >= n_channels) return;

    double attack_coeff = exp(-1.0 / (Fs * (attack * 1e-3)));
    double release_coeff = exp(-1.0 / (Fs * (release * 1e-3)));

    double h = 0.0;  // Initialize filter state for envelope detection

    // Divide samples across threads in parallel
    for (int i = thread_id; i < n_samples; i += stride) {
        int sample_idx = i * n_channels + ch;
        double sample = wav_in[sample_idx];

        // Envelope detection using attack/release dynamics
        double abs_sample = fabs(sample);
        if (abs_sample > h) {
            h = attack_coeff * (h - abs_sample) + abs_sample;
        } else {
            h = release_coeff * (h - abs_sample) + abs_sample;
        }

        // Apply compression/expansion
        double gain;
        if (comp > 0) { // Compression: attenuate higher envelope values
            gain = pow(h + 1e-8, -comp);
        } else { // Expansion: boost lower envelope values
            gain = pow(h + 1e-8, -comp);
        }

        // Scale output
        wav_out[sample_idx] = sample * gain;
    }
}