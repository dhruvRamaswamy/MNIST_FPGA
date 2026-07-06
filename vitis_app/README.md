# MNIST accelerator — Vitis bare-metal app (ZedBoard)

`main.c` streams a 28×28 image to the `myproject_axi` accelerator over AXI DMA
and prints the predicted digit over UART.

## 1. Export hardware from Vivado
After Generate Bitstream:
`File → Export → Export Hardware` → **Include bitstream** → save `design_2_wrapper.xsa`.

## 2. Create the Vitis workspace (Vitis Classic 2023.2)
1. **Tools → Launch Vitis IDE** (or start Vitis), pick a workspace folder.
2. **File → New → Platform Project** → name it `mnist_platform` → **Next** →
   browse to the `.xsa` → OS: **standalone**, CPU: **ps7_cortexa9_0** → Finish.
   Build the platform (hammer icon).
3. **File → New → Application Project** → name `mnist_app` → select the
   `mnist_platform` → **Next** → template **Empty Application (C)** → Finish.
4. Copy this `main.c` into `mnist_app/src/` (replace any existing main).
   - Right-click `mnist_app/src` → Import → File System → select this `main.c`.
5. Build the app (hammer). It should compile clean.

> **Vitis Unified (not Classic)?** There's no `*_DEVICE_ID`. Change the DMA id:
> at the top of `main.c` add `#define DMA_DEV_ID XPAR_XAXIDMA_0_BASEADDR`
> (LookupConfig accepts the base address `0x41E00000` there).

## 3. Run on hardware
1. Connect the ZedBoard USB-UART, set boot mode to **JTAG**, power on.
2. Open a serial terminal (Vitis has one: **Vitis Serial Terminal**) at
   **115200 8N1** on the board's COM port.
3. Right-click `mnist_app` → **Run As → Launch Hardware** (this programs the
   bitstream via the platform, then runs the elf).

## Expected UART output
```
=== MNIST accelerator demo (ZedBoard) ===
  digit 0 : score -1.234
  digit 1 : score -0.567
  ...
  digit 7 : score  6.012     <- largest
  ...
>> Predicted digit: 7   (expected 7)
=== done ===
```

## Notes / gotchas
- **DMA id macro**: if it won't compile, open `xparameters.h` (in the platform's
  `…/include`) and search for `AXI_DMA`. Use whatever `*_DEVICE_ID` it defines.
- **Cache**: the code flushes `tx_buf` before sending and invalidates `rx_buf`
  after receiving — required, since the DMA bypasses the CPU cache.
- **Accuracy caveat**: the host quantizes the input as `ap_fixed<8,4>` (step
  1/16), while training used Brevitas 8-bit (step 0.023). They're close but not
  identical, so a borderline image *could* differ from the 97.5% PyTorch number.
  The bundled "7" is unambiguous and should always read 7.
- **Your own image**: replace `sample_image[]` with raw 0–255 pixel values
  (row-major, 28×28). `preprocess()` handles the normalization + fixed-point
  conversion. White digit on black background, like MNIST.
```
