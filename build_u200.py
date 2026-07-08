"""Build the QAT MNIST model for the AMD Alveo U200 (ORBIT sb9 testbed).

Ported from build3.ipynb (Zedboard / xc7z020) to the VivadoAccelerator
backend, which wraps the HLS IP as a Vitis kernel (.xo), links it with
v++ into an .xclbin for the U200 shell, and emits a PYNQ python driver.

Usage:
  python build_u200.py                # write project + csim accuracy check
  python build_u200.py --skip-csim    # write project only (no torch needed)
  python build_u200.py --build        # + full synthesis and xclbin (needs
                                      #   vivado_hls, vivado, v++ on PATH —
                                      #   run this on the ORBIT node)
  python build_u200.py --build --platform xilinx_u200_gen3x16_xdma_2_202110_1

The --platform value MUST match the shell installed on the node's card:
check with `platforminfo -l` and `xbutil examine` before building.
"""

import argparse
import os
import shutil

import numpy as np
import hls4ml

# hls4ml 1.3.0 bug: InferPrecisionTypes.match() calls self.inputs[0] unconditionally,
# but Constant layers (weights) always have inputs=[]. Patch it to also match
# constants so _infer_default_type() resolves their UnspecifiedPrecisionType.
import hls4ml.model.optimizer.passes.infer_precision as _ip
from hls4ml.model.types import UnspecifiedPrecisionType as _UPT

_orig_ip_match = _ip.InferPrecisionTypes.match


def _safe_ip_match(self, node):
    if not node.inputs:
        return any(isinstance(lt.precision, _UPT) for lt in node.types.values())
    return _orig_ip_match(self, node)


_ip.InferPrecisionTypes.match = _safe_ip_match

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.general import SortGraph

OUTPUT_DIR = 'u200_project'
BOARD = 'alveo-u200'                     # part xcu200-fsgd2104-2-e
CLOCK_PERIOD = 5                         # ns; U200 kernels handle 200 MHz easily
DEFAULT_PRECISION = 'ap_fixed<24,12>'    # same numerics validated on the Zedboard
DEFAULT_REUSE_FACTOR = 4                 # U200 has ~25x the DSPs of the 7z020,
                                         # so trade area back for latency (was 32)


def move_quant_past_maxpool(model):
    """Rewrite Quant -> MaxPool into MaxPool -> Quant (exact: max-pooling
    commutes with monotonic elementwise quantization)."""
    graph = model.graph
    consumers = {}
    for n in graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    changed = False
    for quant in list(graph.node):
        if quant.op_type != 'Quant':
            continue
        cons = consumers.get(quant.output[0], [])
        if len(cons) != 1 or cons[0].op_type != 'MaxPool':
            continue
        pool = cons[0]
        q_in, q_out, p_out = quant.input[0], quant.output[0], pool.output[0]
        mid = q_out + '_prepool'
        pool.input[0] = q_in
        pool.output[0] = mid
        quant.input[0] = mid
        quant.output[0] = p_out
        changed = True
    if changed:
        model = model.transform(SortGraph())
        model = cleanup_model(model)
    return model


def patch_ap_types(project_dir):
    old = ("// #include <complex>\n"
           "namespace std {\n"
           "template<typename _Tp> class complex;\n"
           "}\n")
    for name in ('ap_int_special.h', 'ap_fixed_special.h'):
        path = os.path.join(project_dir, 'firmware', 'ap_types', name)
        with open(path) as f:
            src = f.read()
        if old in src:
            with open(path, 'w') as f:
                f.write(src.replace(old, '#include <complex>\n'))


def prepare_model():
    qonnx_model = ModelWrapper('mnist_finn_ready.onnx')
    qonnx_model = cleanup_model(qonnx_model)
    qonnx_model = qonnx_model.transform(ConvertToChannelsLastAndClean(make_input_channels_last=True))
    qonnx_model = qonnx_model.transform(GemmToMatMul())
    qonnx_model = cleanup_model(qonnx_model)
    qonnx_model = move_quant_past_maxpool(qonnx_model)
    print('Model ready. Ops:', [n.op_type for n in qonnx_model.graph.node])
    return qonnx_model


def make_hls_model(qonnx_model, platform, reuse_factor):
    config = hls4ml.utils.config_from_onnx_model(
        qonnx_model, granularity='name', backend='VivadoAccelerator',
        default_precision=DEFAULT_PRECISION)

    config['Model']['ReuseFactor'] = reuse_factor
    config['Model']['Strategy'] = 'Resource'
    for layer in config.get('LayerName', {}).values():
        layer['ReuseFactor'] = reuse_factor

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    hls_model = hls4ml.converters.convert_from_onnx_model(
        qonnx_model,
        hls_config=config,
        output_dir=OUTPUT_DIR,
        backend='VivadoAccelerator',
        board=BOARD,
        platform=platform,
        interface='axi_stream',
        driver='python',        # emits a PYNQ driver (axi_stream_driver.py)
        clock_period=CLOCK_PERIOD,
        io_type='io_stream',
    )
    hls_model.write()
    print(f'HLS project written to: {OUTPUT_DIR}/')
    return hls_model


def check_accuracy(hls_model):
    import torch  # local import: not needed for --skip-csim / node builds
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    patch_ap_types(OUTPUT_DIR)
    hls_model._compile()  # not .compile(): that re-writes the project and undoes the patch

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081]),
    ])
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=data_transforms)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    images, labels = next(iter(test_loader))
    x = np.ascontiguousarray(images.numpy().transpose(0, 2, 3, 1))  # NCHW -> NHWC
    y = hls_model.predict(x)
    pred = y.reshape(len(labels), 10).argmax(axis=1)
    acc = (pred == labels.numpy()).mean()
    print(f'HLS csim accuracy on {len(labels)} test images: {100 * acc:.2f}%')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true',
                        help='run full synthesis + xclbin link (needs Vivado/Vitis tools)')
    parser.add_argument('--skip-csim', action='store_true',
                        help='skip the C-simulation accuracy check')
    parser.add_argument('--platform', default='xilinx_u200_gen3x16_xdma_2_202110_1',
                        help='Vitis platform of the target card (see `platforminfo -l`)')
    parser.add_argument('--reuse-factor', type=int, default=DEFAULT_REUSE_FACTOR)
    args = parser.parse_args()

    qonnx_model = prepare_model()
    hls_model = make_hls_model(qonnx_model, args.platform, args.reuse_factor)

    if not args.skip_csim:
        check_accuracy(hls_model)

    if args.build:
        # csynth (vivado_hls) -> export IP -> package kernel .xo (vivado)
        # -> v++ link against the U200 shell -> xclbin_files/myproject_kernel.xclbin
        hls_model.build(csim=False, synth=True, export=True, bitfile=True)
        print('Done. xclbin in', os.path.join(OUTPUT_DIR, 'xclbin_files'))


if __name__ == '__main__':
    main()
