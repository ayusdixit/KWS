#include "lstm1.h"
#include "lstm2.h"
#include "dense.h"
#define LSTM2_UNITS 64

#include <ap_fixed.h>
// Increased bit width for better precision
typedef ap_fixed<24,8> fix;  // Changed from 16,8 to 24,8

float cordic_tanh(float theta) {
    #pragma HLS INLINE
    fix theta1 = (fix)theta;
    fix current_cosh = (fix)1.0;
    fix current_sinh = (fix)0;

    fix cordic_phase[10] = {
        (fix)0.549306144,
        (fix)0.255412812,
        (fix)0.125657214,
        (fix)0.062581571,
        (fix)0.031260178,
        (fix)0.015626271,
        (fix)0.007812769,
        (fix)0.003906398,
        (fix)0.001953199,
        (fix)0.000976599
    };

    // Quick return for out-of-range values
    if (theta1 > (fix)1.118) return 0.99999f;
    if (theta1 < (fix)-1.118) return -0.99999f;
    if (theta1 > (fix)-0.01 && theta1 < (fix)0.01) return theta1.to_float();

    fix current_theta = (fix)0;
    fix factor = (fix)0.5;

    CORDIC_ITERATIONS:
    for(int j = 0; j < 10; j++) {
        #pragma HLS UNROLL
        fix sign = (current_theta < theta1) ? (fix)1 : (fix)-1;
        current_theta = current_theta + sign * cordic_phase[j];

        fix cosh_shift = current_cosh * sign * factor;
        fix sinh_shift = current_sinh * sign * factor;

        current_cosh = current_cosh + sinh_shift;
        current_sinh = current_sinh + cosh_shift;

        factor = factor * (fix)0.5;
    }

    fix tanh_result = current_sinh / current_cosh;

    if (tanh_result > (fix)1.0) return 0.99999f;
    if (tanh_result < (fix)-1.0) return -0.99999f;

    return tanh_result.to_float();
}

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
            float g_gate = cordic_tanh(gates[i + 128]);
            float o_gate = sigmoid(gates[i + 192]);

            c_state[i] = f_gate * c_state[i] + i_gate * g_gate;
            h_state[i] = o_gate * cordic_tanh(c_state[i]);
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
            float g_gate = cordic_tanh(gates[i + 128]);
            float o_gate = sigmoid(gates[i + 192]);

            c_state[i] = f_gate * c_state[i] + i_gate * g_gate;
            h_state[i] = o_gate * cordic_tanh(c_state[i]);
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

