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
#include "lstm2.h"
#include "dense.h"
#define LSTM2_UNITS 64


float sigmoid(float x) {
    #pragma HLS INLINE
    return 1.0 / (1.0 + hls::exp(-x));
}


void kws_sequence(
    float input_sequence[49 * INPUT_SIZE],
    float h_final[LSTM1_UNITS],
    float c_final[LSTM1_UNITS],
    float h_states[49][LSTM1_UNITS]
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


        SAVE_H_STATE:
        for (int i = 0; i < LSTM1_UNITS; i++) {
            h_states[t][i] = h_state[i];
        }
    }

    COPY_FINAL:
    for (int i = 0; i < LSTM1_UNITS; i++) {
        h_final[i] = h_state[i];
        c_final[i] = c_state[i];
    }
}

void lstm2_sequence(
    float h_states[49][LSTM1_UNITS],
    float h_final[LSTM2_UNITS],
    float c_final[LSTM2_UNITS]
) {
    float h_state[LSTM2_UNITS];
    float c_state[LSTM2_UNITS];


    INIT_STATES_2:
    for (int i = 0; i < LSTM2_UNITS; i++) {
        h_state[i] = 0;
        c_state[i] = 0;
    }

    float gates[256];


    TIMESTAMP_LOOP_2:
    for (int t = 0; t < 49; t++) {
        #pragma HLS LOOP_TRIPCOUNT min=49 max=49
        float partial_sums[256] = {0};

        INPUT_TRANSFORM_2:
        for (int j = 0; j < LSTM1_UNITS; j++) {
            float input_val = h_states[t][j];
            INNER_INPUT_2:
            for (int i = 0; i < 256; i++) {
                partial_sums[i] += input_val * lstm2_kernel[j][i];
            }
        }


        for (int i = 0; i < 256; i++) {
            gates[i] = partial_sums[i];
        }


        RECURRENT_TRANSFORM_2:
        for (int j = 0; j < LSTM2_UNITS; j++) {
            float h_val = h_state[j];
            INNER_RECURRENT_2:
            for (int i = 0; i < 256; i++) {
                gates[i] += h_val * lstm2_recurrent[j][i];
            }
        }


        BIAS_ADD_2:
        for (int i = 0; i < 256; i++) {
            gates[i] += lstm2_bias[i];
        }


        UPDATE_STATES_2:
        for (int i = 0; i < LSTM2_UNITS; i++) {
            float i_gate = sigmoid(gates[i]);
            float f_gate = sigmoid(gates[i + 64]);
            float g_gate = hls::tanh(gates[i + 128]);
            float o_gate = sigmoid(gates[i + 192]);

            c_state[i] = f_gate * c_state[i] + i_gate * g_gate;
            h_state[i] = o_gate * hls::tanh(c_state[i]);
        }
    }


    COPY_FINAL_2:
    for (int i = 0; i < LSTM2_UNITS; i++) {
        h_final[i] = h_state[i];
        c_final[i] = c_state[i];
    }
}
void softmax(float input[DENSE_UNITS], float output[DENSE_UNITS]) {
    #pragma HLS INLINE off


    float max_val = input[0];
    FIND_MAX:
    for (int i = 1; i < DENSE_UNITS; i++) {
        #pragma HLS PIPELINE
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }


    float exp_sum = 0.0f;
    float exp_values[DENSE_UNITS];

    CALC_EXP:
    for (int i = 0; i < DENSE_UNITS; i++) {
        #pragma HLS PIPELINE
        exp_values[i] = hls::exp(input[i] - max_val);
        exp_sum += exp_values[i];
    }

    NORMALIZE:
    for (int i = 0; i < DENSE_UNITS; i++) {
        #pragma HLS PIPELINE
        output[i] = exp_values[i] / exp_sum;
    }
}


void dense_layer(
    float input[LSTM2_UNITS],
    float output[DENSE_UNITS]
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=input cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=dense_weights cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=dense_bias complete


    float dense_temp[DENSE_UNITS];
    #pragma HLS ARRAY_PARTITION variable=dense_temp complete


    DENSE_LOOP:
    for (int i = 0; i < DENSE_UNITS; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL factor=4

        float acc = 0;
        float bias = dense_bias[i];


        float partial_sums[8] = {0};
        #pragma HLS ARRAY_PARTITION variable=partial_sums complete


        MATRIX_MULT:
        for (int j = 0; j < LSTM2_UNITS; j += 8) {
            #pragma HLS PIPELINE II=1


            float weights_cache[8];
            float inputs_cache[8];
            #pragma HLS ARRAY_PARTITION variable=weights_cache complete
            #pragma HLS ARRAY_PARTITION variable=inputs_cache complete

            CACHE_LOAD:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                if (j + k < LSTM2_UNITS) {
                    weights_cache[k] = dense_weights[j + k][i];
                    inputs_cache[k] = input[j + k];
                }
            }

            // Compute partial products
            PARTIAL_PRODUCTS:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                if (j + k < LSTM2_UNITS) {
                    partial_sums[k] += inputs_cache[k] * weights_cache[k];
                }
            }
        }

        // Sum all partial results
        PARTIAL_SUM:
        for (int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            acc += partial_sums[k];
        }

        dense_temp[i] = acc + bias;
    }


    softmax(dense_temp, output);
}

void lstm_top(
    float input_data[49 * INPUT_SIZE],
    float dense_output[DENSE_UNITS]
) {
    #pragma HLS INTERFACE m_axi port=input_data depth=1960 offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=dense_output depth=4 offset=slave bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=return


    #pragma HLS DATAFLOW


    float h1_state[LSTM1_UNITS];
    float c1_state[LSTM1_UNITS];
    float h2_state[LSTM2_UNITS];
    float c2_state[LSTM2_UNITS];


    float h_states[49][LSTM1_UNITS];

    #pragma HLS ARRAY_PARTITION variable=h_states cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=h1_state cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=c1_state cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=h2_state cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=c2_state cyclic factor=8


    kws_sequence(input_data, h1_state, c1_state, h_states);
    lstm2_sequence(h_states, h2_state, c2_state);
    dense_layer(h2_state, dense_output);
}


