#ifndef SIM_FF_H
#define SIM_FF_H
#include <stdio.h>
#include <stdint.h>
typedef unsigned UINT;
typedef int FRESULT;
typedef struct { int unused; } FATFS;
typedef struct { FILE *handle; uint64_t size, offset; int asset; } FIL;
#define FR_OK 0
#define FA_READ 1
#define FA_WRITE 2
#define FA_OPEN_ALWAYS 16
FRESULT f_mount(FATFS *, const char *, int);
FRESULT f_open(FIL *, const char *, int);
FRESULT f_close(FIL *);
FRESULT f_read(FIL *, void *, UINT, UINT *);
FRESULT f_write(FIL *, const void *, UINT, UINT *);
FRESULT f_sync(FIL *);
FRESULT f_lseek(FIL *, uint64_t);
#define f_size(f) ((f)->size)
#endif
