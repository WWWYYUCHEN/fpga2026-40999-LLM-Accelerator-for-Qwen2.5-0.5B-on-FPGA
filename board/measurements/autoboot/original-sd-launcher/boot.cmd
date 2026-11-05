echo "Qwen25 KV260 100 MHz bare-metal launcher"
setenv qwen_mmc 1
if fatload mmc 1:1 0x10000000 QWEN25.BIN; then
  echo "Using SD mmc 1"
else
  setenv qwen_mmc 0
  if fatload mmc 0:1 0x10000000 QWEN25.BIN; then
    echo "Using SD mmc 0"
  else
    echo "No QWEN25.BIN on SD FAT partition"
    exit 1
  fi
fi
if fpga load 0 0x10000000 ${filesize}; then
  echo "FPGA configured"
else
  echo "FPGA load failed"
  exit 1
fi
if fatload mmc ${qwen_mmc}:1 0x18000000 QWENSD.ELF; then
  bootelf -p 0x18000000
else
  echo "Missing QWENSD.ELF"
  exit 1
fi
