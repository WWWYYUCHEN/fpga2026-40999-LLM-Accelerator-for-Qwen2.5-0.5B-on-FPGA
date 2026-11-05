#include <cstdio>
#include <memory>
#include "verilated.h"
#include "Vlayer.h"

void vl_finish(const char* filename, int line, const char* hierarchy) {
    std::printf("- %s:%d: %s $finish\n", filename, line, hierarchy);
    Verilated::threadContextp()->gotFinish(true);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const auto context = std::make_unique<VerilatedContext>();
    context->threads(1);
    context->commandArgs(argc, argv);
    const auto model = std::make_unique<Vlayer>(context.get(), "");
    std::puts("VERILATOR_SINGLE_LAYER_SINGLE_TOKEN_BEGIN");
    while (!context->gotFinish()) {
        model->eval();
        if (!model->eventsPending()) break;
        context->time(model->nextTimeSlot());
    }
    model->final();
    context->statsPrintSummary();
    return context->gotFinish() ? 0 : 1;
}
