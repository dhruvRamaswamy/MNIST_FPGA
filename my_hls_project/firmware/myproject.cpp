#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &global_in,
    hls::stream<result_t> &layer69_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=global_in,layer69_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<model_default_t, 784>(s79, "s79.txt");
        nnet::load_weights_from_txt<model_default_t, 784>(b79, "b79.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s75, "s75.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b75, "b75.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(s77, "s77.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b77, "b77.txt");
        nnet::load_weights_from_txt<weight97_t, 144>(w97, "w97.txt");
        nnet::load_weights_from_txt<bias97_t, 16>(b97, "b97.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(s93, "s93.txt");
        nnet::load_weights_from_txt<model_default_t, 12544>(b93, "b93.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(s82, "s82.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(b82, "b82.txt");
        nnet::load_weights_from_txt<weight98_t, 2304>(w98, "w98.txt");
        nnet::load_weights_from_txt<bias98_t, 16>(b98, "b98.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(s94, "s94.txt");
        nnet::load_weights_from_txt<model_default_t, 3136>(b94, "b94.txt");
        nnet::load_weights_from_txt<model_default_t, 144>(s85, "s85.txt");
        nnet::load_weights_from_txt<model_default_t, 144>(b85, "b85.txt");
        nnet::load_weights_from_txt<model_default_t, 144>(s86, "s86.txt");
        nnet::load_weights_from_txt<model_default_t, 144>(b86, "b86.txt");
        nnet::load_weights_from_txt<weight95_t, 9216>(w95, "w95.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b95, "b95.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s90, "s90.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b90, "b90.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(s88, "s88.txt");
        nnet::load_weights_from_txt<model_default_t, 64>(b88, "b88.txt");
        nnet::load_weights_from_txt<weight96_t, 640>(w96, "w96.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b96, "b96.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(s91, "s91.txt");
        nnet::load_weights_from_txt<model_default_t, 10>(b91, "b91.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer79_t> layer79_out("layer79_out");
    #pragma HLS STREAM variable=layer79_out depth=784

    hls::stream<layer99_t> layer99_out("layer99_out");
    #pragma HLS STREAM variable=layer99_out depth=900

    hls::stream<layer4_t> Quant_5_param0("Quant_5_param0");
    #pragma HLS STREAM variable=Quant_5_param0 depth=1

    hls::stream<layer5_t> Quant_7_param0("Quant_7_param0");
    #pragma HLS STREAM variable=Quant_7_param0 depth=1

    hls::stream<Quant_5_rescale_result_t> layer75_out("layer75_out");
    #pragma HLS STREAM variable=layer75_out depth=1

    hls::stream<Quant_7_rescale_result_t> layer77_out("layer77_out");
    #pragma HLS STREAM variable=layer77_out depth=1

    hls::stream<Conv2D_Conv_0_result_t> layer97_out("layer97_out");
    #pragma HLS STREAM variable=layer97_out depth=784

    hls::stream<Quant_8_rescale_result_t> layer93_out("layer93_out");
    #pragma HLS STREAM variable=layer93_out depth=784

    hls::stream<layer56_t> layer56_out("layer56_out");
    #pragma HLS STREAM variable=layer56_out depth=784

    hls::stream<layer57_t> layer57_out("layer57_out");
    #pragma HLS STREAM variable=layer57_out depth=196

    hls::stream<layer82_t> layer82_out("layer82_out");
    #pragma HLS STREAM variable=layer82_out depth=196

    hls::stream<layer100_t> layer100_out("layer100_out");
    #pragma HLS STREAM variable=layer100_out depth=256

    hls::stream<Conv2D_Conv_1_result_t> layer98_out("layer98_out");
    #pragma HLS STREAM variable=layer98_out depth=196

    hls::stream<Quant_9_rescale_result_t> layer94_out("layer94_out");
    #pragma HLS STREAM variable=layer94_out depth=196

    hls::stream<layer60_t> layer60_out("layer60_out");
    #pragma HLS STREAM variable=layer60_out depth=196

    hls::stream<layer61_t> layer61_out("layer61_out");
    #pragma HLS STREAM variable=layer61_out depth=9

    hls::stream<layer85_t> layer85_out("layer85_out");
    #pragma HLS STREAM variable=layer85_out depth=9

    hls::stream<Quant_10_rescale_result_t> layer86_out("layer86_out");
    #pragma HLS STREAM variable=layer86_out depth=9

    auto& layer63_out = layer86_out;
    hls::stream<Dense_MatMul_0_result_t> layer95_out("layer95_out");
    #pragma HLS STREAM variable=layer95_out depth=1

    hls::stream<Quant_4_rescale_result_t> layer90_out("layer90_out");
    #pragma HLS STREAM variable=layer90_out depth=1

    hls::stream<Add_0_result_t> layer65_out("layer65_out");
    #pragma HLS STREAM variable=layer65_out depth=1

    hls::stream<layer66_t> layer66_out("layer66_out");
    #pragma HLS STREAM variable=layer66_out depth=1

    hls::stream<layer88_t> layer88_out("layer88_out");
    #pragma HLS STREAM variable=layer88_out depth=1

    hls::stream<Dense_MatMul_1_result_t> layer96_out("layer96_out");
    #pragma HLS STREAM variable=layer96_out depth=1

    hls::stream<Quant_6_rescale_result_t> layer91_out("layer91_out");
    #pragma HLS STREAM variable=layer91_out depth=1

    nnet::normalize<input_t, layer79_t, config79>(global_in, layer79_out, s79, b79); // Quant_8_scale

    nnet::zeropad2d_cl<layer79_t, layer99_t, config99>(layer79_out, layer99_out); // zp2d_Conv2D_Conv_0

    nnet::normalize<layer4_t, Quant_5_rescale_result_t, config75>(Quant_5_param0, layer75_out, s75, b75); // Quant_5_rescale

    nnet::normalize<layer5_t, Quant_7_rescale_result_t, config77>(Quant_7_param0, layer77_out, s77, b77); // Quant_7_rescale

    nnet::conv_2d_cl<layer99_t, Conv2D_Conv_0_result_t, config97>(layer99_out, layer97_out, w97, b97); // Conv2D_Conv_0

    nnet::normalize<Conv2D_Conv_0_result_t, Quant_8_rescale_result_t, config93>(layer97_out, layer93_out, s93, b93); // Quant_8_rescale

    nnet::relu<Quant_8_rescale_result_t, layer56_t, ReLU_config56>(layer93_out, layer56_out); // Relu_0

    nnet::pooling2d_cl<layer56_t, layer57_t, config57>(layer56_out, layer57_out); // MaxPool_0

    nnet::normalize<layer57_t, layer82_t, config82>(layer57_out, layer82_out, s82, b82); // Quant_9_scale

    nnet::zeropad2d_cl<layer82_t, layer100_t, config100>(layer82_out, layer100_out); // zp2d_Conv2D_Conv_1

    nnet::conv_2d_cl<layer100_t, Conv2D_Conv_1_result_t, config98>(layer100_out, layer98_out, w98, b98); // Conv2D_Conv_1

    nnet::normalize<Conv2D_Conv_1_result_t, Quant_9_rescale_result_t, config94>(layer98_out, layer94_out, s94, b94); // Quant_9_rescale

    nnet::relu<Quant_9_rescale_result_t, layer60_t, ReLU_config60>(layer94_out, layer60_out); // Relu_1

    nnet::pooling2d_cl<layer60_t, layer61_t, config61>(layer60_out, layer61_out); // MaxPool_1

    nnet::normalize<layer61_t, layer85_t, config85>(layer61_out, layer85_out, s85, b85); // Quant_10_scale

    nnet::normalize<layer85_t, Quant_10_rescale_result_t, config86>(layer85_out, layer86_out, s86, b86); // Quant_10_rescale

    nnet::dense<Quant_10_rescale_result_t, Dense_MatMul_0_result_t, config95>(layer63_out, layer95_out, w95, b95); // Dense_MatMul_0

    nnet::normalize<Dense_MatMul_0_result_t, Quant_4_rescale_result_t, config90>(layer95_out, layer90_out, s90, b90); // Quant_4_rescale

    nnet::add<Quant_4_rescale_result_t, Quant_5_rescale_result_t, Add_0_result_t, config65>(layer90_out, layer75_out, layer65_out); // Add_0

    nnet::relu<Add_0_result_t, layer66_t, ReLU_config66>(layer65_out, layer66_out); // Relu_2

    nnet::normalize<layer66_t, layer88_t, config88>(layer66_out, layer88_out, s88, b88); // Quant_11_scale

    nnet::dense<layer88_t, Dense_MatMul_1_result_t, config96>(layer88_out, layer96_out, w96, b96); // Dense_MatMul_1

    nnet::normalize<Dense_MatMul_1_result_t, Quant_6_rescale_result_t, config91>(layer96_out, layer91_out, s91, b91); // Quant_6_rescale

    nnet::add<Quant_6_rescale_result_t, Quant_7_rescale_result_t, result_t, config69>(layer91_out, layer77_out, layer69_out); // Add_1

}

