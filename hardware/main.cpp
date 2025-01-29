// best model working right now
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
//    float input[INPUT_SIZE],    // Input vector for timestamp 0
//    float h_out[LSTM1_UNITS],   // Output hidden state (size 64)
//    float c_out[LSTM1_UNITS],   // Output cell state (size 64)
//    float h_prev[LSTM1_UNITS],  // Previous hidden state (size 64)
//    float c_prev[LSTM1_UNITS]   // Previous cell state (size 64)
//) {
//#pragma HLS INTERFACE m_axi port=input depth=40 offset=slave bundle=gmem0
//#pragma HLS INTERFACE m_axi port=h_out depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=c_out depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=h_prev depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE m_axi port=c_prev depth=64 offset=slave bundle=gmem1
//#pragma HLS INTERFACE s_axilite port=return
//
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
//            sum += h_prev[j] * lstm1_recurrent[j][i];
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
//    // Process gates and update states for each unit
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//#pragma HLS PIPELINE
//        // Split gates into input, forget, cell, and output gates
//        float i_gate = sigmoid(gates[i]);                // Input gate
//        float f_gate = sigmoid(gates[i + 64]);          // Forget gate
//        float g_gate = hls::tanh(gates[i + 128]);       // Cell gate
//        float o_gate = sigmoid(gates[i + 192]);         // Output gate
//
//        // Update cell state
//        c_out[i] = f_gate * c_prev[i] + i_gate * g_gate;
//
//        // Update hidden state
//        h_out[i] = o_gate * hls::tanh(c_out[i]);
//    }
//}
///////////////////////////////////////////////////////////////////////////////working fully lsstm 48 times tamp
//#include "lstm1.h"
//
//// Sigmoid activation function
//float sigmoid(float x) {
//    #pragma HLS INLINE
//    return 1.0 / (1.0 + hls::exp(-x));
//}
//
//void kws_sequence(
//    float input_sequence[49 * INPUT_SIZE],
//    float h_final[LSTM1_UNITS],
//    float c_final[LSTM1_UNITS]
//) {
//    #pragma HLS INTERFACE m_axi port=input_sequence depth=1960 offset=slave bundle=gmem0
//    #pragma HLS INTERFACE m_axi port=h_final depth=64 offset=slave bundle=gmem1
//    #pragma HLS INTERFACE m_axi port=c_final depth=64 offset=slave bundle=gmem1
//    #pragma HLS INTERFACE s_axilite port=return
//
//    float h_state[LSTM1_UNITS];
//    float c_state[LSTM1_UNITS];
//
//    // Initialize states
//    INIT_STATES:
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//
//        h_state[i] = 0;
//        c_state[i] = 0;
//    }
//
//    // Local buffer for gates to reduce memory pressure
//    float gates[256];
//
//
//    TIMESTAMP_LOOP:
//    for (int t = 0; t < 49; t++) {
//        #pragma HLS LOOP_TRIPCOUNT min=49 max=49
//
//
//
//        float partial_sums[256] = {0};
//
//
//        INPUT_TRANSFORM:
//        for (int j = 0; j < INPUT_SIZE; j++) {
//
//            int input_index = t * INPUT_SIZE + j;
//            float input_val = input_sequence[input_index];
//
//            INNER_INPUT:
//            for (int i = 0; i < 256; i++) {
//
//                partial_sums[i] += input_val * lstm1_kernel[j][i];
//            }
//        }
//
//        // Copy partial sums to gates
//        COPY_PARTIAL:
//        for (int i = 0; i < 256; i++) {
//
//            gates[i] = partial_sums[i];
//        }
//
//        // Recurrent transformation: h_state * recurrent_kernel
//        RECURRENT_TRANSFORM:
//        for (int j = 0; j < LSTM1_UNITS; j++) {
//
//            float h_val = h_state[j];
//
//            INNER_RECURRENT:
//            for (int i = 0; i < 256; i++) {
//
//                gates[i] += h_val * lstm1_recurrent[j][i];
//            }
//        }
//
//        // Add bias
//        BIAS_ADD:
//        for (int i = 0; i < 256; i++) {
//
//            gates[i] += lstm1_bias[i];
//        }
//
//        // Process gates and update states
//        UPDATE_STATES:
//        for (int i = 0; i < LSTM1_UNITS; i++) {
//
//
//            // Split gates
//            float i_gate = sigmoid(gates[i]);
//            float f_gate = sigmoid(gates[i + 64]);
//            float g_gate = hls::tanh(gates[i + 128]);
//            float o_gate = sigmoid(gates[i + 192]);
//
//            // Update cell state
//            c_state[i] = f_gate * c_state[i] + i_gate * g_gate;
//
//            // Update hidden state
//            h_state[i] = o_gate * hls::tanh(c_state[i]);
//        }
//    }
//
//    // Copy final states to output
//    COPY_FINAL:
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//
//        h_final[i] = h_state[i];
//        c_final[i] = c_state[i];
//    }
//}

///////////////

#include "lstm1.h"

// Original sigmoid function
float sigmoid(float x) {
    #pragma HLS INLINE
    return 1.0 / (1.0 + hls::exp(-x));
}

// Original kws_sequence function
void kws_sequence(
    float input_sequence[49 * INPUT_SIZE],
    float h_final[LSTM1_UNITS],
    float c_final[LSTM1_UNITS]
) {


    float h_state[LSTM1_UNITS];
    float c_state[LSTM1_UNITS];

    INIT_STATES:
    for (int i = 0; i < LSTM1_UNITS; i++) {
        h_state[i] = 0;
        c_state[i] = 0;
    }

    float gates[256];

    TIMESTAMP_LOOP:
    for (int t = 0; t < 49; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=49 max=49
        float partial_sums[256] = {0};

        INPUT_TRANSFORM:
        for (int j = 0; j < INPUT_SIZE; j++) {
            int input_index = t * INPUT_SIZE + j;
            float input_val = input_sequence[input_index];
            INNER_INPUT:
            for (int i = 0; i < 256; i++) {
                partial_sums[i] += input_val * lstm1_kernel[j][i];
            }
        }

        COPY_PARTIAL:
        for (int i = 0; i < 256; i++) {
            gates[i] = partial_sums[i];
        }

        RECURRENT_TRANSFORM:
        for (int j = 0; j < LSTM1_UNITS; j++) {
            float h_val = h_state[j];
            INNER_RECURRENT:
            for (int i = 0; i < 256; i++) {
                gates[i] += h_val * lstm1_recurrent[j][i];
            }
        }

        BIAS_ADD:
        for (int i = 0; i < 256; i++) {
            gates[i] += lstm1_bias[i];
        }

        UPDATE_STATES:
        for (int i = 0; i < LSTM1_UNITS; i++) {
            float i_gate = sigmoid(gates[i]);
            float f_gate = sigmoid(gates[i + 64]);
            float g_gate = hls::tanh(gates[i + 128]);
            float o_gate = sigmoid(gates[i + 192]);

            c_state[i] = f_gate * c_state[i] + i_gate * g_gate;
            h_state[i] = o_gate * hls::tanh(c_state[i]);
        }
    }

    COPY_FINAL:
    for (int i = 0; i < LSTM1_UNITS; i++) {
        h_final[i] = h_state[i];
        c_final[i] = c_state[i];
    }
}

// Top function
void lstm_top(
    float input_data[49 * INPUT_SIZE],
    float output_h[LSTM1_UNITS],
    float output_c[LSTM1_UNITS]
) {
    #pragma HLS INTERFACE m_axi port=input_data depth=1960 offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output_h depth=64 offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output_c depth=64 offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return

    // Call the kws_sequence function
    kws_sequence(input_data, output_h, output_c);
}
