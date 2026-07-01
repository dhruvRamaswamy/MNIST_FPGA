/*
 * MNIST accelerator demo - ZedBoard (Zynq-7020) bare-metal
 * Vitis Unified IDE 2023.2 (SDT flow -> XAxiDma_LookupConfig uses BASE ADDRESS).
 *
 * Streams a 28x28 image to the hls4ml accelerator over AXI DMA, reads back the
 * 10 class scores, argmax = predicted digit. The accelerator is free-running
 * (ap_ctrl_none) so there is no start bit -- just push data through the DMA.
 *
 * Encodings:
 *   INPUT  x_1        : ap_fixed<8,4>  -> byte = round(value*16),
 *                       value = (pixel/255 - 0.1307)/0.3081  (training Normalize)
 *   OUTPUT layer25_out: result_t = ap_fixed<23,15>  (23-bit, 8 fractional bits).
 *                       The wrapper zero-fills bit 23, so the SIGN bit is bit 22.
 *                       score = sign_extend_from_bit22(word) / 256.0
 */

#include <stdio.h>
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"

#define IMG_SIZE   784      /* 28*28 input bytes  */
#define NUM_CLASS  10       /* 10 output words    */

/* Output fixed-point format (must match result_t = ap_fixed<W,I>). */
#define OUT_W       23      /* total bits          */
#define OUT_FRAC    8       /* W - I = 23 - 15     */
#define OUT_SCALE   (1 << OUT_FRAC)      /* 256    */
#define OUT_SIGNBIT (1 << (OUT_W - 1))   /* bit 22 */
#define OUT_MASK    ((1 << OUT_W) - 1)   /* low 23 bits */

/* SDT / Vitis Unified: LookupConfig takes the DMA BASE ADDRESS. */
#define DMA_BASEADDR  XPAR_AXI_DMA_0_BASEADDR   /* 0x41E00000 */

static XAxiDma AxiDma;

static int8_t  tx_buf[IMG_SIZE]  __attribute__((aligned(64)));
static int32_t rx_buf[NUM_CLASS] __attribute__((aligned(64)));

/* MNIST test image #0 (a "7"), raw 0..255 pixels. */
static const unsigned char sample_image[IMG_SIZE] = {
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0, 84,185,159,151, 60, 36,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,222,254,254,254,254,241,198,198,198,198,198,198,198,198,170, 52,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0, 67,114, 72,114,163,227,254,225,254,254,254,250,229,254,254,140,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 17, 66, 14, 67, 67, 67, 59, 21,236,254,106,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 83,253,209, 18,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 22,233,255, 83,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,129,254,238, 44,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 59,249,254, 62,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,133,254,187,  5,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,205,248, 58,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,126,254,182,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 75,251,240, 57,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 19,221,254,166,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  3,203,254,219, 35,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 38,254,254, 77,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 31,224,254,115,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,133,254,254, 52,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 61,242,254,254, 52,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,121,254,254,219, 40,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,121,254,207, 18,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
};

/* raw pixel -> ap_fixed<8,4> byte, applying the training Normalize. */
static void preprocess(const unsigned char *img, int8_t *out)
{
    int i;
    for (i = 0; i < IMG_SIZE; i++) {
        float v = (((float)img[i] / 255.0f) - 0.1307f) / 0.3081f;
        float s = v * 16.0f;                              /* ap_fixed<8,4>: 2^4 */
        int   q = (int)(s + (s >= 0.0f ? 0.5f : -0.5f));  /* round, no libm */
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

/* Decode one output word: sign-extend the OUT_W-bit value (sign at bit OUT_W-1). */
static int32_t decode_score(int32_t w)
{
    w &= OUT_MASK;                       /* keep the OUT_W significant bits    */
    if (w & OUT_SIGNBIT)                 /* sign bit set -> extend to 32 bits  */
        w |= ~OUT_MASK;
    return w;                            /* raw fixed-point (÷256 = real value)*/
}

static int init_dma(void)
{
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_BASEADDR);
    if (!cfg) { xil_printf("No DMA config @ 0x%08x\r\n", DMA_BASEADDR); return XST_FAILURE; }
    if (XAxiDma_CfgInitialize(&AxiDma, cfg) != XST_SUCCESS) {
        xil_printf("DMA init failed\r\n"); return XST_FAILURE;
    }
    if (XAxiDma_HasSg(&AxiDma)) { xil_printf("DMA in SG mode!\r\n"); return XST_FAILURE; }
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    return XST_SUCCESS;
}

int main(void)
{
    int c;
    xil_printf("\r\n=== MNIST accelerator demo (ZedBoard) ===\r\n");

    if (init_dma() != XST_SUCCESS) return XST_FAILURE;

    preprocess(sample_image, tx_buf);

    Xil_DCacheFlushRange((UINTPTR)tx_buf, IMG_SIZE);
    Xil_DCacheFlushRange((UINTPTR)rx_buf, sizeof(rx_buf));

    /* arm receive first (accelerator free-runs) */
    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)rx_buf, NUM_CLASS * sizeof(int32_t),
                               XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
        xil_printf("S2MM start failed\r\n"); return XST_FAILURE;
    }
    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)tx_buf, IMG_SIZE,
                               XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
        xil_printf("MM2S start failed\r\n"); return XST_FAILURE;
    }

    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA) ||
           XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* poll */ }

    Xil_DCacheInvalidateRange((UINTPTR)rx_buf, sizeof(rx_buf));

    int     best_i = 0;
    int32_t best_v = decode_score(rx_buf[0]);
    for (c = 0; c < NUM_CLASS; c++) {
        int32_t raw = decode_score(rx_buf[c]);           /* signed fixed-point */
        int     neg = (raw < 0);
        int32_t a   = neg ? -raw : raw;
        int     ip  = a / OUT_SCALE;                     /* integer part       */
        int     fp  = ((a % OUT_SCALE) * 1000) / OUT_SCALE;  /* 3 decimals     */
        xil_printf("  digit %d : %s%d.%03d\r\n", c, neg ? "-" : "", ip, fp);
        if (raw > best_v) { best_v = raw; best_i = c; }
    }

    xil_printf("-----------------------------------------\r\n");
    xil_printf(">> Predicted digit: %d   (expected 7)\r\n", best_i);
    xil_printf("=== done ===\r\n");
    return XST_SUCCESS;
}
