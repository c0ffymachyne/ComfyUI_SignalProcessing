#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C" __global__
void limiter_kernel(
    const double*__restrict__ wav_in,       // Input audio signal [L0, R0, L1, R1, ..., LN-1, RN-1]
    double* __restrict__ wav_out,            // Output audio signal
    double* __restrict__ debug_out,          // Debug buffer [envelope, gain]
    const int n_channels,       // Number of channels (e.g., 2 for stereo)
    const int n_samples,        // Number of samples per channel
    double threshold,     // Threshold in percents (0-100)
    double slope,         // Slope angle in percents (0-100)
    const double sr,            // Sample rate (samples/sec)
    double twnd,          // Window time for RMS in ms
    double tatt,          // Attack time in ms
    double trel           // Release time in ms
) {
    // Only one thread handles the entire stereo pair
    int ch = blockIdx.x * blockDim.x + threadIdx.x; // Thread processes a single channel

    if (ch >= n_channels) return;

    double attack_coeff = exp(-1.0 / (sr * (tatt * 1e-3)));
    double release_coeff = exp(-1.0 / (sr * (trel * 1e-3)));
    double envelope = 0.00;
    //threshold = .55;
    //slope = 1.0;

    for (int i = 0; i < n_samples; ++i) {

        double sample = wav_in[i * n_channels + ch];

        // Envelope tracking
        double abs_sample = fabs(sample);
        if (abs_sample > envelope) {
            envelope = attack_coeff * (envelope - abs_sample) + abs_sample;
        } else {
            envelope = release_coeff * (envelope - abs_sample) + abs_sample;
        }

        // Gain calculation
        double gain = 1.0;
        if (envelope > threshold) {
            gain = pow(10.0, -slope * (log10(envelope) - log10(threshold)));
        }
        // Upward compression below threshold
        double upward_compression_gain = 1.0;
        if (envelope < threshold && envelope > 0.0) {
            upward_compression_gain = pow(10.0, slope * (log10(threshold) - log10(envelope)));
        }
        // Apply gain
        wav_out[i * n_channels + ch] = sample * gain * upward_compression_gain;

        // Debugging output (envelope and gain)
        if (debug_out) {
            debug_out[i * n_channels + ch] = gain;
        }
    }
}