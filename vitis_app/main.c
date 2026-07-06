/*
 * MNIST accelerator demo - ZedBoard (Zynq-7020) bare-metal
 * --------------------------------------------------------
 * Sends a 28x28 image to the hls4ml accelerator over AXI DMA and reads back
 * the 10 class scores.
 *
 * Datapath (built in Vivado):
 *   PS DDR  --MM2S-->  AXI DMA  --AXIS(8b)-->  myproject_axi.in_r
 *   myproject_axi.out_r  --AXIS(32b)-->  AXI DMA  --S2MM-->  PS DDR
 *
 * The accelerator is free-running (ap_ctrl_none): no start bit, just stream
 * data through the DMA.
 *
 * Encodings (must match how the model was trained / exported):
 *   INPUT  x_1        : ap_fixed<8,4>  -> send signed byte = round(value * 16)
 *                       value = (pixel/255 - 0.1307) / 0.3081   (training Normalize)
 *   OUTPUT layer25_out: ap_fixed<24,16> in the low 24 bits of each 32-bit word
 *                       score = sign_extend24(word) / 256.0
 *   argmax over the 10 scores = predicted digit.
 */

#include <stdio.h>
#include <math.h>
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_printf.h"

#define IMG_SIZE   784   /* 28 * 28 input pixels (bytes)      */
#define NUM_CLASS  10    /* 10 output class scores (words)    */

/* ----- AXI DMA device id -----------------------------------------------
 * Check your xparameters.h for the exact macro. In Vitis Classic it is
 * usually XPAR_AXI_DMA_0_DEVICE_ID (sometimes XPAR_AXIDMA_0_DEVICE_ID).
 * In Vitis Unified (SDT) there is no DEVICE_ID -- use the base address
 * 0x41E00000 with XAxiDma_LookupConfig(XPAR_XAXIDMA_0_BASEADDR) instead.
 */
#ifndef DMA_DEV_ID
#define DMA_DEV_ID  XPAR_AXI_DMA_0_DEVICE_ID
#endif

static XAxiDma AxiDma;

/* DMA buffers in DDR. Aligned to a cache line (32 B on Cortex-A9) so the
 * flush/invalidate ranges don't disturb neighbouring data. */
static int8_t  tx_buf[IMG_SIZE]  __attribute__((aligned(64)));
static int32_t rx_buf[NUM_CLASS] __attribute__((aligned(64)));

/* A real MNIST test image (index 0, a "7"), raw 0..255 pixels. */
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

/* Convert raw 0..255 pixels into ap_fixed<8,4> signed bytes the accelerator
 * expects, applying the same Normalize(mean=0.1307, std=0.3081) used in
 * training. */
static void preprocess(const unsigned char *img, int8_t *out)
{
    for (int i = 0; i < IMG_SIZE; i++) {
        float v = (((float)img[i] / 255.0f) - 0.1307f) / 0.3081f;
        int   q = (int)lroundf(v * 16.0f);   /* ap_fixed<8,4>: scale by 2^4 */
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}

/* Sign-extend the low 24 bits of an ap_fixed<24,16> word, return raw int. */
static int32_t sext24(int32_t w)
{
    w &= 0x00FFFFFF;
    if (w & 0x00800000) w |= 0xFF000000;   /* extend bit 23 */
    return w;
}

static int init_dma(void)
{
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(DMA_DEV_ID);
    if (!cfg) { xil_printf("No DMA config for id %d\r\n", DMA_DEV_ID); return XST_FAILURE; }
    if (XAxiDma_CfgInitialize(&AxiDma, cfg) != XST_SUCCESS) {
        xil_printf("DMA init failed\r\n"); return XST_FAILURE;
    }
    if (XAxiDma_HasSg(&AxiDma)) {            /* we built it in simple mode */
        xil_printf("DMA is in SG mode - rebuild AXI DMA with SG disabled\r\n");
        return XST_FAILURE;
    }
    /* no interrupts - we poll */
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    XAxiDma_IntrDisable(&AxiDma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    return XST_SUCCESS;
}

int main(void)
{
    xil_printf("\r\n=== MNIST accelerator demo (ZedBoard) ===\r\n");

    if (init_dma() != XST_SUCCESS) return XST_FAILURE;

    preprocess(sample_image, tx_buf);

    /* Push tx_buf out of cache so the DMA sees it; make rx region coherent. */
    Xil_DCacheFlushRange((UINTPTR)tx_buf, IMG_SIZE);
    Xil_DCacheFlushRange((UINTPTR)rx_buf, sizeof(rx_buf));

    /* Arm the receive (S2MM) FIRST -- the accelerator free-runs and will emit
     * the result as soon as it has consumed the input. */
    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)rx_buf, NUM_CLASS * sizeof(int32_t),
                               XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS) {
        xil_printf("S2MM start failed\r\n"); return XST_FAILURE;
    }
    /* Then send the image (MM2S). */
    if (XAxiDma_SimpleTransfer(&AxiDma, (UINTPTR)tx_buf, IMG_SIZE,
                               XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS) {
        xil_printf("MM2S start failed\r\n"); return XST_FAILURE;
    }

    /* Poll until both channels are idle. (Add a timeout in production code.) */
    while (XAxiDma_Busy(&AxiDma, XAXIDMA_DEVICE_TO_DMA) ||
           XAxiDma_Busy(&AxiDma, XAXIDMA_DMA_TO_DEVICE)) { /* spin */ }

    /* Invalidate so we read what the DMA wrote, not stale cache. */
    Xil_DCacheInvalidateRange((UINTPTR)rx_buf, sizeof(rx_buf));

    /* Decode the 10 scores and argmax. xil_printf has no %f, so print the
     * raw fixed-point score and a scaled integer (score * 1000). */
    int     best_i = 0;
    int32_t best_v = sext24(rx_buf[0]);
    for (int c = 0; c < NUM_CLASS; c++) {
        int32_t raw = sext24(rx_buf[c]);
        int     milli = (raw * 1000) / 256;       /* score * 1000 */
        xil_printf("  digit %d : score %d.%03d%s\r\n",
                   c, milli / 1000, (milli < 0 ? -milli : milli) % 1000,
                   (c == 0 ? "" : ""));
        if (raw > best_v) { best_v = raw; best_i = c; }
    }

    xil_printf("-----------------------------------------\r\n");
    xil_printf(">> Predicted digit: %d   (expected 7)\r\n", best_i);
    xil_printf("=== done ===\r\n");
    return XST_SUCCESS;
}
