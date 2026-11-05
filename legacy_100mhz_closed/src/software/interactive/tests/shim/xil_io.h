#ifndef SIM_XIL_IO_H
#define SIM_XIL_IO_H
#include <stdint.h>
typedef uintptr_t UINTPTR;
void Xil_Out64(UINTPTR, uint64_t);
uint64_t Xil_In64(UINTPTR);
#endif
