#ifndef QWEN_GENERATION_H
#define QWEN_GENERATION_H
#include <stdint.h>
/* All model computations are started through CSR callbacks. */
struct qgen_ops {
    void *context;
    void (*write)(void *, unsigned, uint64_t);
    uint64_t (*read)(void *, unsigned);
    uint64_t (*ticks)(void *);
    void (*wait)(void *);
    void (*embedding)(void *, unsigned token, unsigned row);
    int (*token)(void *, unsigned iteration, unsigned token, uint64_t elapsed);
    uint64_t ticks_per_second;
};
struct qgen_result { unsigned produced; int eos; uint64_t last_status; };
/* -1 bad input, -2 timeout, -3 PL error, -4 invalid token, -5 output error. */
int qgen_run(const struct qgen_ops *ops, const uint32_t *prompt, unsigned count,
             unsigned max_new, struct qgen_result *result);
#endif
