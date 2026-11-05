#ifndef QWEN_UTF8_STREAM_H
#define QWEN_UTF8_STREAM_H
#include <stddef.h>
typedef void (*qutf8_sink)(const unsigned char *, size_t, void *);
struct qutf8_stream { unsigned char pending[4]; unsigned used, expected; qutf8_sink sink; void *context; };
void qutf8_init(struct qutf8_stream *s, qutf8_sink sink, void *context);
void qutf8_push(struct qutf8_stream *s, const unsigned char *bytes, size_t length);
void qutf8_finish(struct qutf8_stream *s);
#endif
