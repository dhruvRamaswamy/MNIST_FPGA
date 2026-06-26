#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<8,4>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,4>, 1*1> layer28_t;
typedef ap_fixed<21,13> Conv2D_conv1_Conv_accum_t;
typedef nnet::array<ap_fixed<21,13>, 32*1> Conv2D_conv1_Conv_result_t;
typedef ap_fixed<8,4> model_default_t;
typedef nnet::array<ap_fixed<8,4>, 32*1> layer11_t;
typedef ap_fixed<18,8> relu1_act_quant_activation_impl_Relu_table_t;
typedef ap_fixed<8,4> pool_MaxPool_accum_t;
typedef nnet::array<ap_fixed<8,4>, 32*1> layer12_t;
typedef nnet::array<ap_fixed<8,4>, 32*1> layer29_t;
typedef ap_fixed<26,18> Conv2D_conv2_Conv_accum_t;
typedef nnet::array<ap_fixed<26,18>, 64*1> Conv2D_conv2_Conv_result_t;
typedef nnet::array<ap_fixed<8,4>, 64*1> layer14_t;
typedef ap_fixed<18,8> relu2_act_quant_activation_impl_Relu_table_t;
typedef ap_fixed<8,4> pool2_MaxPool_accum_t;
typedef nnet::array<ap_fixed<8,4>, 64*1> layer15_t;
typedef ap_fixed<27,19> Dense_fc1_Gemm_matmul_accum_t;
typedef nnet::array<ap_fixed<27,19>, 128*1> Dense_fc1_Gemm_matmul_result_t;
typedef ap_uint<1> layer24_index;
typedef nnet::array<ap_fixed<8,4>, 128*1> layer19_t;
typedef ap_fixed<18,8> relu3_act_quant_activation_impl_Relu_table_t;
typedef ap_fixed<24,16> Dense_fc2_Gemm_matmul_accum_t;
typedef nnet::array<ap_fixed<24,16>, 10*1> result_t;
typedef ap_uint<1> layer25_index;

// hls-fpga-machine-learning insert emulator-defines


#endif
