#include "generation.h"
#include <string.h>
#define CAPACITY 256U
#define VOCAB 151936U
int qgen_run(const struct qgen_ops *ops, const uint32_t *prompt, unsigned count,
             unsigned max_new, struct qgen_result *result) {
    memset(result, 0, sizeof(*result));
    if (!count || !max_new || max_new >= CAPACITY || count > CAPACITY - max_new) return -1;
    for (unsigned i = 0; i < count; ++i) if (prompt[i] >= VOCAB) return -1;
    for (unsigned row = 0; row < count; ++row) ops->embedding(ops->context, prompt[row], row);
    unsigned position = 0, rows = count;
    for (unsigned iteration = 0; iteration < max_new; ++iteration) {
        ops->write(ops->context, 0x0080, iteration != 0);
        ops->write(ops->context, 0x0088, rows);
        ops->write(ops->context, 0x0090, position + rows);
        ops->write(ops->context, 0x0098, position);
        uint64_t start = ops->ticks(ops->context), now;
        ops->write(ops->context, 0x0010, 1);
        for (;;) {
            result->last_status = ops->read(ops->context, 0x0018);
            now = ops->ticks(ops->context);
            if (result->last_status & (1ULL << 3)) return -3;
            if ((result->last_status & 0x46U) == 0x44U) break;
            /* Long full-model RTL can exceed the old 120-second timeout. */
            if (now - start > 1800ULL * ops->ticks_per_second) return -2;
            ops->wait(ops->context);
        }
        uint64_t value = ops->read(ops->context, 0x01f8);
        if (value >= VOCAB) return -4;
        unsigned token = (unsigned)value;
        if (ops->token(ops->context, iteration, token, now - start)) return -5;
        ++result->produced;
        if (token == 151643 || token == 151645) { result->eos = 1; break; }
        position += rows;
        rows = 1;
        if (iteration + 1 < max_new) ops->embedding(ops->context, token, 0);
    }
    return 0;
}
