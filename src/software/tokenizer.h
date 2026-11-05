#ifndef QWEN_TOKENIZER_H
#define QWEN_TOKENIZER_H
#include <stddef.h>
#include <stdint.h>
#define QT_TEXT_LIMIT 4096
#define QT_FILE_LIMIT (8U * 1024U * 1024U)
/* Buffer must stay alive and be 4-byte aligned. Returns zero on success. */
int qt_init(const void *data, size_t bytes);
int qt_encode(const char *utf8, uint32_t *ids, unsigned capacity, unsigned *count);
int qt_chat_encode(const char *utf8, uint32_t *ids, unsigned capacity, unsigned *count);
/* Token fragments are raw UTF-8 bytes and may end inside a codepoint. */
int qt_decode(uint32_t id, const unsigned char **bytes, unsigned *length, int *special);
unsigned qt_vocab_size(void);
const char *qt_error(void);
#endif
