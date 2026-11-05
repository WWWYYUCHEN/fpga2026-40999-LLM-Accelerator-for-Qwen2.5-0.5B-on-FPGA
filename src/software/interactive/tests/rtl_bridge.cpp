/* Reuse the frozen Verilated accelerator. Replace only its test stimulus. */
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include "verilated.h"
#include "Vlayer.h"
#include "Vlayer___024root.h"
#include "svdpi.h"

#define SIGNAL(name) (model->rootp->tb_qwen25_inference__DOT__##name)
static std::unique_ptr<VerilatedContext> context;
static std::unique_ptr<Vlayer> model;
static std::map<uint64_t, std::vector<uint8_t>> memory;
static std::vector<uint8_t> embedding;
static std::ofstream logits, tokens;
static float maximum;
static unsigned maximum_id, logit_count, iteration, produced;
static uint64_t iteration_start;
static std::string sd_root;
static bool layer_diagnostic;
static bool previous_layer_done;
static bool layer_trace;
static unsigned configured_position;
static unsigned configured_rows=1;
static std::vector<unsigned> prompt_ids;
static std::ofstream norm_probe, norm_input_probe, weight_probe, activation_probe;
static std::ofstream lm_norm_probe;
static uint64_t ignored_idle_bias_addresses;
extern "C" bool qwen_bias_bounds_used(unsigned enabled) {
    if (!enabled) ++ignored_idle_bias_addresses;
    return enabled != 0;
}

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot read " + path);
    std::vector<uint8_t> data(static_cast<size_t>(in.tellg()));
    in.seekg(0); in.read(reinterpret_cast<char *>(data.data()), data.size());
    if (!in) throw std::runtime_error("Short read " + path);
    return data;
}
static uint8_t *locate(uint64_t address, size_t bytes) {
    auto it = memory.upper_bound(address);
    if (it == memory.begin()) throw std::runtime_error("Unmapped DDR address");
    --it;
    uint64_t offset = address - it->first;
    if (offset > it->second.size() || bytes > it->second.size() - offset) {
        std::fprintf(stderr, "UNMAPPED DDR %llx size=%zu\n", static_cast<unsigned long long>(address), bytes);
        throw std::runtime_error("DDR range outside allocated segment");
    }
    return it->second.data() + offset;
}
extern "C" void qwen_ddr_read(uint64_t address, svBitVecVal *out) { std::memcpy(out, locate(address, 16), 16); }
extern "C" void qwen_ddr_write(uint64_t address, const svBitVecVal *in) { std::memcpy(locate(address, 16), in, 16); }
extern "C" int qwen_prompt_rows() { return 40; } // Unused: autonomous test stimulus is disabled.
extern "C" void qwen_dump_layer() { }
extern "C" void qwen_prepare_input(int token, int) {
    std::memcpy(locate(0x71000000, 1792), embedding.data() + static_cast<size_t>(token) * 1792, 1792);
}
extern "C" void qwen_begin_iteration(int index) {
    maximum = -std::numeric_limits<float>::infinity(); maximum_id = 0; logit_count = 0;
    logits.close(); logits.open("hardware_logits_" + std::to_string(index) + ".bin", std::ios::binary | std::ios::trunc);
    if (!logits) throw std::runtime_error("Cannot record logits");
}
extern "C" void qwen_observe_logits(int address, const svBitVecVal *data) {
    if (static_cast<unsigned>(address) * 8 != logit_count) throw std::runtime_error("Logits order mismatch");
    uint16_t values[8]; std::memcpy(values, data, 16);
    logits.write(reinterpret_cast<const char *>(values), 16);
    for (uint16_t bf16 : values) {
        if ((bf16 & 0x7f80) == 0x7f80) throw std::runtime_error("Nonfinite logit");
        uint32_t bits = static_cast<uint32_t>(bf16) << 16;
        float value; std::memcpy(&value, &bits, 4);
        if (value >= maximum) { maximum = value; maximum_id = logit_count; }
        ++logit_count;
    }
}
extern "C" void qwen_report_token(int index, int token, int cycles) {
    logits.flush();
    lm_norm_probe.flush();
    if (logit_count != 151936 || static_cast<unsigned>(token) != maximum_id) throw std::runtime_error("CSR greedy mismatch");
    if (SIGNAL(commits) != 1 || SIGNAL(norm_words) != 112 || SIGNAL(final_starts) != 1 || SIGNAL(lm_commands) != 132944)
        throw std::runtime_error("RTL completion coverage mismatch");
    tokens << index << ',' << token << ',' << cycles << '\n'; tokens.flush();
    std::printf("\nRTL_TOKEN_VERIFIED iteration=%d id=%d cycles=%d logits=%u max=%g idle_bias_addresses=%llu\n", index, token, cycles, logit_count, maximum, static_cast<unsigned long long>(ignored_idle_bias_addresses));
    ++produced;
}
static void step_event() {
    if (!model->eventsPending() || context->gotFinish()) throw std::runtime_error("RTL stopped unexpectedly");
    if(layer_trace && !SIGNAL(clk) && SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__act_norm_result) &&
       SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__operation)==7)
        lm_norm_probe.write(reinterpret_cast<const char *>(SIGNAL(dut__DOT__afu_Inst__DOT__result_data).data()),16);
    if(layer_diagnostic && !SIGNAL(clk)) {
        if(SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__act_norm_result))
            norm_probe.write(reinterpret_cast<const char *>(SIGNAL(dut__DOT__afu_Inst__DOT__result_data).data()),16);
        if(SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__gemm_act_valid) && SIGNAL(dut__DOT__s_gemm_act_ready))
            activation_probe.write(reinterpret_cast<const char *>(SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__activation).data()),128);
    }
    context->time(model->nextTimeSlot()); model->eval();
    if (layer_diagnostic || layer_trace) {
        bool done = ((SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__state_current)[2] >> 5) & 1U) != 0;
        if (done && !previous_layer_done) {
            unsigned layer=SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__layer);
            std::string prefix=layer_diagnostic ? "layer0_" : "iteration_"+std::to_string(iteration)+"_layer_"+std::to_string(layer)+"_";
            if(layer==0) {
                std::ofstream gamma_dump("gamma_ram.bin",std::ios::binary|std::ios::trunc);
                for(unsigned word=0;word<5488;++word) {
                    uint64_t halves[]={
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__gamma_Inst__DOT__mem0_Inst__DOT__memory_Inst__DOT__memory)[word],
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__gamma_Inst__DOT__mem1_Inst__DOT__memory_Inst__DOT__memory)[word]};
                    gamma_dump.write(reinterpret_cast<char *>(halves),16);
                }
                std::ofstream bias_dump("bias_ram.bin",std::ios::binary|std::ios::trunc);
                for(unsigned word=0;word<3456;++word) {
                    uint64_t halves[]={
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__bias_Inst__DOT__mem0_Inst__DOT__memory_Inst__DOT__memory)[word],
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__bias_Inst__DOT__mem1_Inst__DOT__memory_Inst__DOT__memory)[word]};
                    bias_dump.write(reinterpret_cast<char *>(halves),16);
                }
                std::ofstream scale_dump("scale_ram.bin",std::ios::binary|std::ios::trunc);
                for(unsigned word=0;word<28504;++word) {
                    uint64_t quarters[]={
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__scale_Inst__DOT__mem0_Inst__DOT__memory_Inst__DOT__memory)[word],
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__scale_Inst__DOT__mem1_Inst__DOT__memory_Inst__DOT__memory)[word],
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__scale_Inst__DOT__mem2_Inst__DOT__memory_Inst__DOT__memory)[word],
                        SIGNAL(dut__DOT__mmu_Inst__DOT__controller_Inst__DOT__buffers_Inst__DOT__scale_Inst__DOT__mem3_Inst__DOT__memory_Inst__DOT__memory)[word]};
                    scale_dump.write(reinterpret_cast<char *>(quarters),32);
                }
            }
            struct Dump { const char *name; uint64_t base; size_t bytes; };
            const Dump dumps[] = {{"input",0x71000000,1792},{"k",0x10000000+uint64_t(layer)*0x800000+uint64_t(configured_position)*256,256},{"v",0x20000000+uint64_t(layer)*0x800000+uint64_t(configured_position)*256,256},
                {"q",0x71500000,1792},{"context",0x71600000,1792},{"h",0x71700000+uint64_t(configured_position)*1792,1792},
                {"m",0x71800000+uint64_t(configured_position)*9728,9728},{"y",0x71b00000+uint64_t(configured_position)*1792,1792}};
            for (auto &d : dumps) {
                std::ofstream out(prefix + d.name + ".bin",std::ios::binary);
                size_t rows=std::min(configured_rows,8U);
                out.write(reinterpret_cast<char *>(locate(d.base,d.bytes*rows)),d.bytes*rows);
            }
            std::ofstream ids("prompt_ids.txt");
            for(unsigned i=0;i<configured_rows && i<prompt_ids.size();++i) ids<<prompt_ids[i]<<' ';
            ids<<'\n'; ids.close();
            if(layer_diagnostic) {
                std::printf("LAYER_DIAGNOSTIC_COMPLETE cycles=%u\n",SIGNAL(cycles));
                norm_probe.flush(); weight_probe.flush(); activation_probe.flush();
                std::fflush(stdout); std::exit(0);
            }
        }
        previous_layer_done = done;
    }
}
static void edge(bool high) { do step_event(); while (static_cast<bool>(SIGNAL(clk)) != high); }
extern "C" void rtl_advance(unsigned cycles) {
    for (unsigned i = 0; i < cycles; ++i) { edge(true); edge(false); }
}
extern "C" uint64_t rtl_ticks() { return SIGNAL(cycles) / 100; }
extern "C" void rtl_clear() {
    // Clear only mutable regions; weights and descriptors remain loaded.
    for (auto &item : memory) if (item.first < 0x40000000 || item.first >= 0x71000000)
        std::fill(item.second.begin(), item.second.end(), 0);
    iteration = 0;
}
extern "C" void rtl_embedding(unsigned token, unsigned row) {
    if (token >= 151936 || row >= 256) throw std::runtime_error("Embedding bounds");
    std::memcpy(locate(0x71000000 + static_cast<uint64_t>(row) * 1792, 1792),
                embedding.data() + static_cast<size_t>(token) * 1792, 1792);
    if(prompt_ids.size()<=row)prompt_ids.resize(row+1);
    prompt_ids[row]=token;
}
extern "C" void rtl_write(unsigned address, uint64_t value) {
    if (address == 0x88) { SIGNAL(test_rows) = static_cast<unsigned>(value); configured_rows=static_cast<unsigned>(value); }
    if (address == 0x98) configured_position=static_cast<unsigned>(value);
    if (address == 0x80 && value == 0) iteration = 0;
    if (address == 0x10 && value == 1) {
        SIGNAL(iteration) = iteration;
        SIGNAL(layer_completions) = SIGNAL(final_starts) = SIGNAL(logits_words) = 0;
        SIGNAL(norm_words) = SIGNAL(lm_commands) = SIGNAL(commits) = 0;
        iteration_start = SIGNAL(cycles);
        qwen_begin_iteration(static_cast<int>(iteration));
    }
    edge(false);
    SIGNAL(s_axil_awaddr) = address; SIGNAL(s_axil_wdata) = value; SIGNAL(s_axil_wstrb) = 255;
    SIGNAL(s_axil_awvalid) = SIGNAL(s_axil_wvalid) = SIGNAL(s_axil_bready) = 1;
    model->eval();
    bool aw = false, w = false;
    unsigned guard = 0;
    while (!aw || !w) {
        if (++guard > 1000) throw std::runtime_error("AXIL write handshake timeout");
        bool accept_aw = SIGNAL(s_axil_awvalid) && SIGNAL(s_axil_awready);
        bool accept_w = SIGNAL(s_axil_wvalid) && SIGNAL(s_axil_wready);
        edge(true); aw |= accept_aw; w |= accept_w; edge(false);
        SIGNAL(s_axil_awvalid) = !aw; SIGNAL(s_axil_wvalid) = !w; model->eval();
    }
    while (!SIGNAL(dut__DOT__mmu_Inst__DOT__registers_Inst__DOT__transport_Inst__DOT__bvalid_o)) { if (++guard > 1000) throw std::runtime_error("AXIL B timeout"); edge(false); }
    if (SIGNAL(dut__DOT__mmu_Inst__DOT__registers_Inst__DOT__transport_Inst__DOT__bresp_o)) throw std::runtime_error("AXIL write error response");
    edge(true); edge(false); SIGNAL(s_axil_bready) = 0; model->eval();
}
extern "C" uint64_t rtl_read(unsigned address) {
    edge(false); SIGNAL(s_axil_araddr) = address; SIGNAL(s_axil_arvalid) = SIGNAL(s_axil_rready) = 1; model->eval();
    unsigned guard = 0;
    for (;;) {
        bool accept = SIGNAL(s_axil_arready); edge(true); edge(false);
        if (accept) break;
        if (++guard > 1000) throw std::runtime_error("AXIL AR timeout");
    }
    SIGNAL(s_axil_arvalid) = 0; model->eval();
    while (!SIGNAL(dut__DOT__mmu_Inst__DOT__registers_Inst__DOT__transport_Inst__DOT__rvalid_o)) { if (++guard > 1000) throw std::runtime_error("AXIL R timeout"); edge(false); }
    if (SIGNAL(dut__DOT__mmu_Inst__DOT__registers_Inst__DOT__transport_Inst__DOT__rresp_o)) throw std::runtime_error("AXIL read error response");
    uint64_t value = SIGNAL(dut__DOT__mmu_Inst__DOT__registers_Inst__DOT__transport_Inst__DOT__rdata_o);
    edge(true); edge(false); SIGNAL(s_axil_rready) = 0; model->eval();
    if (address == 0x1f8) {
        qwen_report_token(static_cast<int>(iteration++), static_cast<int>(value), static_cast<int>(SIGNAL(cycles) - iteration_start));
    }
    return value;
}
void vl_finish(const char *filename, int line, const char *) {
    std::fprintf(stderr, "Unexpected $finish at %s:%d\n", filename, line);
    context->gotFinish(true);
}
extern "C" int qwen_firmware_main(void);
int main(int argc, char **argv) {
    try {
        const char *root = std::getenv("SIM_SD"); if (!root) throw std::runtime_error("SIM_SD required"); sd_root = root;
        layer_diagnostic = std::getenv("RTL_LAYER_DIAGNOSTIC") != nullptr;
        layer_trace=std::getenv("RTL_DUMP_LAYERS") != nullptr;
        if(layer_trace)lm_norm_probe.open("lm_norm_probe.bin",std::ios::binary|std::ios::trunc);
        if(layer_diagnostic) {
            norm_probe.open("norm_probe.bin",std::ios::binary|std::ios::trunc);
            activation_probe.open("activation_probe.bin",std::ios::binary|std::ios::trunc);
            weight_probe.open("weight_probe.bin",std::ios::binary|std::ios::trunc);
        }
        std::ifstream regions("segments.tsv"); std::string base, name; size_t bytes;
        while (regions >> base >> bytes >> name) {
            auto data = name == "-" ? std::vector<uint8_t>(bytes, 0) : read_file(sd_root + "/" + name);
            if (data.size() != bytes) throw std::runtime_error("Asset size mismatch " + name);
            memory.emplace(std::stoull(base, nullptr, 16), std::move(data));
        }
        if (memory.empty()) throw std::runtime_error("segments.tsv missing");
        embedding = read_file(sd_root + "/QWEN25/EMBED.BIN");
        tokens.open("hardware_tokens.csv", std::ios::trunc); tokens << "iteration,token_id,cycles\n";
        context = std::make_unique<VerilatedContext>(); context->threads(1); context->commandArgs(argc, argv);
        model = std::make_unique<Vlayer>(context.get(), ""); model->eval();
        SIGNAL(rst_n) = 0;
        rtl_advance(8); SIGNAL(rst_n) = 1; rtl_advance(8);
        int status = qwen_firmware_main();
        model->final();
        return status;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "RTL_BRIDGE_ERROR %s\n", error.what()); return 2;
    }
}
