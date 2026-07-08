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
typedef nnet::array<ap_fixed<24,12>, 1*1> input_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 1*1> layer79_t;
typedef ap_fixed<24,12> model_default_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 1*1> layer99_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 64*1> layer4_t;
typedef nnet::array<ap_fixed<8,8,AP_RND_CONV,AP_SAT,0>, 10*1> layer5_t;
typedef nnet::array<ap_fixed<33,21>, 64*1> Quant_5_rescale_result_t;
typedef nnet::array<ap_fixed<33,21>, 10*1> Quant_7_rescale_result_t;
typedef ap_fixed<17,17> Conv2D_Conv_0_accum_t;
typedef nnet::array<ap_fixed<17,17>, 16*1> Conv2D_Conv_0_result_t;
typedef ap_fixed<4,4,AP_RND_CONV,AP_SAT_SYM,0> weight97_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT,0> bias97_t;
typedef nnet::array<ap_fixed<42,30>, 16*1> Quant_8_rescale_result_t;
typedef nnet::array<ap_fixed<24,12>, 16*1> layer56_t;
typedef ap_fixed<18,8> Relu_0_table_t;
typedef ap_fixed<24,12> MaxPool_0_accum_t;
typedef nnet::array<ap_fixed<24,12>, 16*1> layer57_t;
typedef nnet::array<ap_ufixed<2,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer82_t;
typedef nnet::array<ap_ufixed<2,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer100_t;
typedef ap_fixed<13,13> Conv2D_Conv_1_accum_t;
typedef nnet::array<ap_fixed<13,13>, 16*1> Conv2D_Conv_1_result_t;
typedef ap_fixed<2,2,AP_RND_CONV,AP_SAT_SYM,0> weight98_t;
typedef ap_fixed<8,8,AP_RND_CONV,AP_SAT,0> bias98_t;
typedef nnet::array<ap_fixed<38,26>, 16*1> Quant_9_rescale_result_t;
typedef nnet::array<ap_fixed<24,12>, 16*1> layer60_t;
typedef ap_fixed<18,8> Relu_1_table_t;
typedef ap_fixed<24,12> MaxPool_1_accum_t;
typedef nnet::array<ap_fixed<24,12>, 16*1> layer61_t;
typedef nnet::array<ap_ufixed<2,2,AP_RND_CONV,AP_SAT,0>, 16*1> layer85_t;
typedef nnet::array<ap_fixed<27,15>, 16*1> Quant_10_rescale_result_t;
typedef ap_fixed<38,26> Dense_MatMul_0_accum_t;
typedef nnet::array<ap_fixed<38,26>, 64*1> Dense_MatMul_0_result_t;
typedef ap_fixed<2,2,AP_RND_CONV,AP_SAT_SYM,0> weight95_t;
typedef ap_uint<1> layer95_index;
typedef nnet::array<ap_fixed<63,39>, 64*1> Quant_4_rescale_result_t;
typedef nnet::array<ap_fixed<64,40>, 64*1> Add_0_result_t;
typedef nnet::array<ap_fixed<24,12>, 64*1> layer66_t;
typedef ap_fixed<18,8> Relu_2_table_t;
typedef nnet::array<ap_ufixed<2,2,AP_RND_CONV,AP_SAT,0>, 64*1> layer88_t;
typedef ap_fixed<25,13> Dense_MatMul_1_accum_t;
typedef nnet::array<ap_fixed<25,13>, 10*1> Dense_MatMul_1_result_t;
typedef ap_fixed<4,4,AP_RND_CONV,AP_SAT_SYM,0> weight96_t;
typedef ap_uint<1> layer96_index;
typedef nnet::array<ap_fixed<50,26>, 10*1> Quant_6_rescale_result_t;
typedef nnet::array<ap_fixed<51,27>, 10*1> result_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
