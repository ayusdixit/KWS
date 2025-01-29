//#include "lstm1.h"
//#include <stdio.h>
//#include <float.h>
//#include <math.h>
//
//// Function prototype
//void kws(float input[INPUT_SIZE], float h_out[LSTM1_UNITS],
//         float c_out[LSTM1_UNITS], float h_prev[LSTM1_UNITS],
//         float c_prev[LSTM1_UNITS]);
//
//int main() {
//    // Initialize input and state arrays
//    float local_input[INPUT_SIZE];  // Input features
//    float h_out[LSTM1_UNITS];       // Output hidden state
//    float c_out[LSTM1_UNITS];       // Output cell state
//    float h_prev[LSTM1_UNITS];      // Previous hidden state
//    float c_prev[LSTM1_UNITS];      // Previous cell state
//
//    // Initialize input with some test values
//    for (int i = 0; i < INPUT_SIZE; i++) {
//        local_input[i] = 0;  // Simple test pattern
//    }
//
//    // Initialize previous states to zero
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//        h_prev[i] = 0.0f;
//        c_prev[i] = 0.0f;
//    }
//
//    // Print some debug information for input
//    printf("Input array (first 10 elements):\n");
//    for (int i = 0; i < 10; i++) {
//        printf("input[%d] = %f\n", i, local_input[i]);
//    }
//
//    // Print initial states
//    printf("\nInitial hidden state (first 10 elements):\n");
//    for (int i = 0; i < 10; i++) {
//        printf("h_prev[%d] = %f\n", i, h_prev[i]);
//    }
//
//    // Call the LSTM function
//    kws(local_input, h_out, c_out, h_prev, c_prev);
//
//    // Print the output hidden state
//    printf("\nOutput hidden state (first 10 elements):\n");
//    for (int i = 0; i < 10; i++) {
//        if (isnan(h_out[i]) || isinf(h_out[i]) || fabs(h_out[i]) > FLT_MAX) {
//            printf("Error: h_out[%d] = %f (Invalid value)\n", i, h_out[i]);
//        } else {
//            printf("h_out[%d] = %f\n", i, h_out[i]);
//        }
//    }
//
//    // Print the output cell state
//    printf("\nOutput cell state (first 10 elements):\n");
//    for (int i = 0; i < 10; i++) {
//        if (isnan(c_out[i]) || isinf(c_out[i]) || fabs(c_out[i]) > FLT_MAX) {
//            printf("Error: c_out[%d] = %f (Invalid value)\n", i, c_out[i]);
//        } else {
//            printf("c_out[%d] = %f\n", i, c_out[i]);
//        }
//    }
//
//    // Check for any invalid values in both output states
//    int invalid_count = 0;
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//        if (isnan(h_out[i]) || isinf(h_out[i]) || fabs(h_out[i]) > FLT_MAX) {
//            invalid_count++;
//        }
//        if (isnan(c_out[i]) || isinf(c_out[i]) || fabs(c_out[i]) > FLT_MAX) {
//            invalid_count++;
//        }
//    }
//
//    if (invalid_count > 0) {
//        printf("\nWarning: %d invalid values found in the output states.\n", invalid_count);
//    } else {
//        printf("\nAll output values are valid.\n");
//    }
//
//    printf("Execution completed\n");
//    return 0;
//}

//////////////////////////////////////////////////////testbnehc working
//#include "lstm1.h"
//#include <stdio.h>
//#include <float.h>
//#include <math.h>
//
//// Function prototype for the LSTM sequence processing function
//void kws_sequence(float input_sequence[49 * INPUT_SIZE], float h_final[LSTM1_UNITS], float c_final[LSTM1_UNITS]);
//
//int main() {
//    // Declare arrays for final hidden and cell states
//    float h_final[LSTM1_UNITS];  // Final hidden state output
//    float c_final[LSTM1_UNITS];  // Final cell state output
//
//    // Initialize final states to zero
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//        h_final[i] = 0.0f;
//        c_final[i] = 0.0f;
//    }
//
//    // Call the LSTM sequence processing function
//    // Use the pre-defined `input` array from lstm1.h
//    kws_sequence(input, h_final, c_final);
//
//    // Print final outputs for verification
//    printf("\n=== Final Hidden State ===\n");
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//        printf("h_final[%d] = %f\n", i, h_final[i]);
//    }
//
//    printf("\n=== Final Cell State ===\n");
//    for (int i = 0; i < LSTM1_UNITS; i++) {
//        printf("c_final[%d] = %f\n", i, c_final[i]);
//    }
//
//    return 0;
//}

///////////////////////////////////////////////////////////////////////////////////////////////
#include "lstm1.h"
#include <stdio.h>
#include <float.h>
#include <math.h>

// Function prototype for the top-level LSTM function
void lstm_top(float input_sequence[49 * INPUT_SIZE], float h_final[LSTM1_UNITS], float c_final[LSTM1_UNITS]);

int main() {
    // Declare arrays for final hidden and cell states
    float h_final[LSTM1_UNITS];  // Final hidden state output
    float c_final[LSTM1_UNITS];  // Final cell state output

    // Initialize final states to zero
    for (int i = 0; i < LSTM1_UNITS; i++) {
        h_final[i] = 0.0f;
        c_final[i] = 0.0f;
    }

    // Call the top-level LSTM function
    // Use the pre-defined `input` array from lstm1.h
    lstm_top(input, h_final, c_final);

    // Print final outputs for verification
    printf("\n=== Final Hidden State ===\n");
    for (int i = 0; i < LSTM1_UNITS; i++) {
        printf("h_final[%d] = %f\n", i, h_final[i]);
    }

    printf("\n=== Final Cell State ===\n");
    for (int i = 0; i < LSTM1_UNITS; i++) {
        printf("c_final[%d] = %f\n", i, c_final[i]);
    }

    return 0;
}
