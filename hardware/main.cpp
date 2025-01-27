//#include "lstm1.h"
//
//// Sigmoid activation function
//float sigmoid(float x) {
//#pragma HLS INLINE
//    return 1.0 / (1.0 + hls::exp(-x));
//}
//
//// LSTM1 Cell for timestamp 0
//void kws(
//    float input[INPUT_SIZE],  // Input vector for timestamp 0
//    float &h_out,             // Output hidden state
//    float &c_out,             // Output cell state
//    float h_prev,             // Previous hidden state (initialized to 0)
//    float c_prev              // Previous cell state (initialized to 0)
//) {
//
//#pragma HLS INTERFACE m_axi port=input depth=40 offset=slave bundle=gmem0
//#pragma HLS INTERFACE m_axi port=h_out depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=c_out depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=h_prev depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=c_prev depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE s_axilite port=return
//    // Gates: input, forget, cell, output
//    float gates[256];
//
//    // Input transformation: input * kernel
//    for (int i = 0; i < 256; i++) {
//#pragma HLS PIPELINE
//        float sum = 0;
//        for (int j = 0; j < INPUT_SIZE; j++) {
//            sum += input[j] * lstm1_kernel[j][i];
//        }
//        gates[i] = sum;
//    }
//
//    // Recurrent transformation: h_prev * recurrent_kernel
//    for (int i = 0; i < 256; i++) {
//#pragma HLS PIPELINE
//        float sum = 0;
//        for (int j = 0; j < LSTM1_UNITS; j++) {
//            sum += h_prev * lstm1_recurrent[j][i];
//        }
//        gates[i] += sum;
//    }
//
//    // Add bias
//    for (int i = 0; i < 256; i++) {
//#pragma HLS PIPELINE
//        gates[i] += lstm1_bias[i];
//    }
//
//    // Split gates into input, forget, cell, and output gates
//    float i_gate = sigmoid(gates[0]);          // Input gate
//    float f_gate = sigmoid(gates[64]);         // Forget gate
//    float g_gate = hls::tanh(gates[128]);      // Cell gate
//    float o_gate = sigmoid(gates[192]);        // Output gate
//
//    // Update cell state: c_out = f_gate * c_prev + i_gate * g_gate
//    c_out = f_gate * c_prev + i_gate * g_gate;
//
//    // Update hidden state: h_out = o_gate * tanh(c_out)
//    h_out = o_gate * hls::tanh(c_out);
//}

#include "lstm1.h"

// Sigmoid activation function
float sigmoid(float x) {
#pragma HLS INLINE
    return 1.0 / (1.0 + hls::exp(-x));
}

// LSTM1 Cell for timestamp 0
void kws(
    float input[INPUT_SIZE],    // Input vector for timestamp 0
    float h_out[LSTM1_UNITS],   // Output hidden state (size 64)
    float c_out[LSTM1_UNITS],   // Output cell state (size 64)
    float h_prev[LSTM1_UNITS],  // Previous hidden state (size 64)
    float c_prev[LSTM1_UNITS]   // Previous cell state (size 64)
) {
#pragma HLS INTERFACE m_axi port=input depth=40 offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=h_out depth=64 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=c_out depth=64 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=h_prev depth=64 offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=c_prev depth=64 offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=return

    // Gates: input, forget, cell, output
    float gates[256];

    // Input transformation: input * kernel
    for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE
        float sum = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += input[j] * lstm1_kernel[j][i];
        }
        gates[i] = sum;
    }

    // Recurrent transformation: h_prev * recurrent_kernel
    for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE
        float sum = 0;
        for (int j = 0; j < LSTM1_UNITS; j++) {
            sum += h_prev[j] * lstm1_recurrent[j][i];
        }
        gates[i] += sum;
    }

    // Add bias
    for (int i = 0; i < 256; i++) {
#pragma HLS PIPELINE
        gates[i] += lstm1_bias[i];
    }

    // Process gates and update states for each unit
    for (int i = 0; i < LSTM1_UNITS; i++) {
#pragma HLS PIPELINE
        // Split gates into input, forget, cell, and output gates
        float i_gate = sigmoid(gates[i]);                // Input gate
        float f_gate = sigmoid(gates[i + 64]);          // Forget gate
        float g_gate = hls::tanh(gates[i + 128]);       // Cell gate
        float o_gate = sigmoid(gates[i + 192]);         // Output gate

        // Update cell state
        c_out[i] = f_gate * c_prev[i] + i_gate * g_gate;

        // Update hidden state
        h_out[i] = o_gate * hls::tanh(c_out[i]);
    }
}
