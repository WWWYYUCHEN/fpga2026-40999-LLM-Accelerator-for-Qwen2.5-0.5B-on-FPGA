echo "Qwen25 KV260 AUTOBOOT QWUART 2026-10-25"
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
if fatload mmc ${qwen_mmc}:1 0x18000000 QWUART.ELF; then
  bootelf -p 0x18000000
else
  echo "Missing QWUART.ELF"
  exit 1
fi
