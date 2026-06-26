#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &x_1,
    hls::stream<result_t> &layer25_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x_1,layer25_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 288>(w26, "w26.txt");
        nnet::load_weights_from_txt<model_default_t, 32>(b26, "b26.txt");
        nnet::load_weights_from_txt<model_default_t, 18432>(w27, "w27.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b27, "b27.txt");
        nnet::load_weights_from_txt<model_default_t, 73728>(w24, "w24.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(b24, "b24.txt");
        nnet::load_weights_from_txt<model_default_t, 1280>(w25, "w25.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b25, "b25.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=900

    hls::stream<Conv2D_conv1_Conv_result_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=784

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=784

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=196

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=256

    hls::stream<Conv2D_conv2_Conv_result_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=196

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=196

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=9

    auto& layer16_out = layer15_out;
    hls::stream<Dense_fc1_Gemm_matmul_result_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=1

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=1

    nnet::zeropad2d_cl<input_t, layer28_t, config28>(x_1, layer28_out); // zp2d_Conv2D_conv1_Conv

    nnet::conv_2d_cl<layer28_t, Conv2D_conv1_Conv_result_t, config26>(layer28_out, layer26_out, w26, b26); // Conv2D_conv1_Conv

    nnet::relu<Conv2D_conv1_Conv_result_t, layer11_t, ReLU_config11>(layer26_out, layer11_out); // relu1_act_quant_activation_impl_Relu

    nnet::pooling2d_cl<layer11_t, layer12_t, config12>(layer11_out, layer12_out); // pool_MaxPool

    nnet::zeropad2d_cl<layer12_t, layer29_t, config29>(layer12_out, layer29_out); // zp2d_Conv2D_conv2_Conv

    nnet::conv_2d_cl<layer29_t, Conv2D_conv2_Conv_result_t, config27>(layer29_out, layer27_out, w27, b27); // Conv2D_conv2_Conv

    nnet::relu<Conv2D_conv2_Conv_result_t, layer14_t, ReLU_config14>(layer27_out, layer14_out); // relu2_act_quant_activation_impl_Relu

    nnet::pooling2d_cl<layer14_t, layer15_t, config15>(layer14_out, layer15_out); // pool2_MaxPool

    nnet::dense<layer15_t, Dense_fc1_Gemm_matmul_result_t, config24>(layer16_out, layer24_out, w24, b24); // Dense_fc1_Gemm_matmul

    nnet::relu<Dense_fc1_Gemm_matmul_result_t, layer19_t, ReLU_config19>(layer24_out, layer19_out); // relu3_act_quant_activation_impl_Relu

    nnet::dense<layer19_t, result_t, config25>(layer19_out, layer25_out, w25, b25); // Dense_fc2_Gemm_matmul

}

