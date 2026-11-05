#include <stdint.h>
typedef struct { uintptr_t BaseAddress; } XUartPs_Config;
typedef struct { int unused; } XUartPs;
#define XUARTPS_OPER_MODE_NORMAL 0
XUartPs_Config *XUartPs_LookupConfig(uintptr_t);
int XUartPs_CfgInitialize(XUartPs *, XUartPs_Config *, uintptr_t);
int XUartPs_SetBaudRate(XUartPs *, unsigned);
void XUartPs_SetOperMode(XUartPs *, unsigned);
void XUartPs_SendByte(uintptr_t, unsigned char);
unsigned char XUartPs_RecvByte(uintptr_t);
