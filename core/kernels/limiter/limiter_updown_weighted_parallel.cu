#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" __global__
void limiter_kernel(
    const double* __restrict__ wav_in,       // Input audio signal [L0, R0, L1, R1, ..., LN-1, RN-1]
    double* __restrict__ wav_out,            // Output audio signal
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

    double attack_coeff = exp(-1.0 / (sr * (tatt * 1e-3)));
    double release_coeff = exp(-1.0 / (sr * (trel * 1e-3)));

    // Use release time to determine precompute samples
    int precompute_samples = int((trel * 1e-3) * sr); // Convert release time to samples

    // Each thread computes its own range of samples
    int start_idx = thread_id * (n_samples / stride);
    int end_idx = (thread_id + 1) * (n_samples / stride);

    // Extend range backward for precomputing
    int precompute_start_idx = max(0, start_idx - precompute_samples);

    double envelope = 0.0;  // Envelope tracking state

    // Precompute envelope for the extra range
    for (int i = precompute_start_idx; i < start_idx; ++i) {
        int sample_idx = i * n_channels + ch;
        double sample = wav_in[sample_idx];
        double abs_sample = fabs(sample);

        if (abs_sample > envelope) {
            envelope = attack_coeff * (envelope - abs_sample) + abs_sample;
        } else {
            envelope = release_coeff * (envelope - abs_sample) + abs_sample;
        }
    }

    // Process assigned range of samples
    for (int i = start_idx; i < end_idx; ++i) {
        int sample_idx = i * n_channels + ch;
        double sample = wav_in[sample_idx];
        double abs_sample = fabs(sample);

        // Envelope tracking
        if (abs_sample > envelope) {
            envelope = attack_coeff * (envelope - abs_sample) + abs_sample;
        } else {
            envelope = release_coeff * (envelope - abs_sample) + abs_sample;
        }

        // Gain calculation for downward limiting
        double gain = 1.0;
        if (envelope > threshold) {
            gain = pow(10.0, -slope * (log10(envelope) - log10(threshold)));
        }

        // Upward compression below threshold with gradual application
        double upward_compression_gain = 1.0;
        if (envelope < threshold && envelope > 0.0) {
            double t = 1.0 - threshold; // Scaling factor based on threshold
            // Dynamic factor increases as envelope decreases
            double dynamic_factor = (threshold - envelope) / threshold;
            // Clamp to [0, 1]
            dynamic_factor = fmin(fmax(dynamic_factor, 0.0), 1.0);
            upward_compression_gain = 1.0 + t * dynamic_factor * pow(10.0, slope * (log10(threshold) - log10(envelope)));
        }

        // Apply both gains
        wav_out[sample_idx] = sample * gain * upward_compression_gain;
    }
}