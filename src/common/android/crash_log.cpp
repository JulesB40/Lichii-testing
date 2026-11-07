// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/prctl.h>
#include <ucontext.h>
#include <unistd.h>

#include "common/android/crash_log.h"
#include "common/signal_chain.h"

namespace Common::Android {

namespace {

constexpr std::array FatalSignals{SIGILL, SIGSEGV, SIGBUS, SIGFPE, SIGABRT};

char g_path[1024];
struct sigaction g_previous[NSIG];
volatile sig_atomic_t g_reported;

// Formats without allocating: the handler runs on a crashed thread.
class Writer {
public:
    void Str(const char* text) {
        while (*text != '\0' && size < buffer.size()) {
            buffer[size++] = *text++;
        }
    }

    void Hex(std::uint64_t value) {
        Str("0x");
        char digits[16];
        int count = 0;
        do {
            digits[count++] = "0123456789abcdef"[value & 0xf];
            value >>= 4;
        } while (value != 0);
        while (count > 0 && size < buffer.size()) {
            buffer[size++] = digits[--count];
        }
    }

    void Dec(std::int64_t value) {
        if (value < 0) {
            Str("-");
            value = -value;
        }
        char digits[20];
        int count = 0;
        do {
            digits[count++] = static_cast<char>('0' + value % 10);
            value /= 10;
        } while (value != 0);
        while (count > 0 && size < buffer.size()) {
            buffer[size++] = digits[--count];
        }
    }

    void WriteTo(int fd) const {
        std::size_t done = 0;
        while (done < size) {
            const ssize_t result = write(fd, buffer.data() + done, size - done);
            if (result <= 0) {
                return;
            }
            done += static_cast<std::size_t>(result);
        }
    }

private:
    std::array<char, 8192> buffer;
    std::size_t size = 0;
};

const char* SignalName(int sig) {
    switch (sig) {
    case SIGILL:
        return "SIGILL";
    case SIGSEGV:
        return "SIGSEGV";
    case SIGBUS:
        return "SIGBUS";
    case SIGFPE:
        return "SIGFPE";
    case SIGABRT:
        return "SIGABRT";
    default:
        return "signal";
    }
}

#ifdef __aarch64__
// NCE points TPIDR_EL0 at its native context while guest code runs; the context holds this magic
// at offset 0x20 (core/arm/nce/arm_nce_asm_definitions.h).
bool InGuestCode() {
    std::uint64_t tpidr;
    asm volatile("mrs %0, tpidr_el0" : "=r"(tpidr));
    if (tpidr == 0 || (tpidr & 3) != 0) {
        return false;
    }
    return *reinterpret_cast<const volatile std::uint32_t*>(tpidr + 0x20) == 0x555a5559;
}

// Host return addresses may carry a pointer authentication code above the address bits.
bool Describe(Writer& out, std::uint64_t address) {
    for (const std::uint64_t mask : {~std::uint64_t{0}, (std::uint64_t{1} << 48) - 1,
                                     (std::uint64_t{1} << 39) - 1}) {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(address & mask), &info) != 0 &&
            info.dli_fname != nullptr) {
            const char* name = std::strrchr(info.dli_fname, '/');
            out.Str(" ");
            out.Str(name != nullptr ? name + 1 : info.dli_fname);
            out.Str("+");
            out.Hex((address & mask) - reinterpret_cast<std::uint64_t>(info.dli_fbase));
            if (info.dli_sname != nullptr) {
                out.Str(" (");
                out.Str(info.dli_sname);
                out.Str(")");
            }
            return true;
        }
    }
    return false;
}

// ART turns SIGSEGV and SIGBUS in managed code (implicit null checks, stack overflow) into Java
// exceptions; those pass through unreported.
bool IsManagedCode(std::uint64_t pc) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(pc), &info) == 0 || info.dli_fname == nullptr) {
        return true; // JIT code, or no image at all
    }
    for (const char* runtime : {"libart", ".oat", ".odex", ".art", ".vdex"}) {
        if (std::strstr(info.dli_fname, runtime) != nullptr) {
            return true;
        }
    }
    return false;
}
#endif

void Report(int sig, siginfo_t* info, void* raw_context) {
    auto* context = static_cast<ucontext_t*>(raw_context);
    Writer out;
    out.Str("\n[ crash ] Fatal ");
    out.Str(SignalName(sig));
    out.Str(" code ");
    out.Dec(info->si_code);
    out.Str(", fault address ");
    out.Hex(reinterpret_cast<std::uint64_t>(info->si_addr));
    char thread_name[17]{};
    prctl(PR_GET_NAME, thread_name, 0, 0, 0);
    out.Str(", thread ");
    out.Str(thread_name);
    out.Str("\n");
#ifdef __aarch64__
    const auto& mc = context->uc_mcontext;
    const bool guest = InGuestCode();
    out.Str("[ crash ] pc ");
    out.Hex(mc.pc);
    if (guest) {
        out.Str(" in guest code (find its module in the \"Module ... loaded at\" lines)");
    } else if (!Describe(out, mc.pc)) {
        out.Str(" (no image)");
    }
    const bool pc_faulted = reinterpret_cast<std::uint64_t>(info->si_addr) - mc.pc < 4;
    if ((mc.pc & 3) == 0 && !(sig == SIGSEGV && pc_faulted)) {
        out.Str(", instruction ");
        out.Hex(*reinterpret_cast<const volatile std::uint32_t*>(mc.pc));
    }
    out.Str("\n[ crash ] lr ");
    out.Hex(mc.regs[30]);
    if (!guest) {
        Describe(out, mc.regs[30]);
    }
    out.Str(", sp ");
    out.Hex(mc.sp);
    out.Str(", pstate ");
    out.Hex(mc.pstate);
    for (int i = 0; i < 29; ++i) {
        out.Str(i % 4 == 0 ? "\n[ crash ] " : "  ");
        out.Str("x");
        out.Dec(i);
        out.Str(" ");
        out.Hex(mc.regs[i]);
    }
    out.Str("\n[ crash ] fp ");
    out.Hex(mc.regs[29]);
    // Frame records while they stay on this stack: 16 bytes apart at least, within 8 MiB.
    std::uint64_t fp = mc.regs[29];
    for (int frame = 0; frame < 24; ++frame) {
        if (fp == 0 || (fp & 7) != 0 || fp < mc.sp || fp - mc.sp > (8U << 20)) {
            break;
        }
        const auto* record = reinterpret_cast<const volatile std::uint64_t*>(fp);
        const std::uint64_t next = record[0];
        const std::uint64_t return_address = record[1];
        out.Str("\n[ crash ]   #");
        out.Dec(frame);
        out.Str(" ");
        out.Hex(return_address);
        if (!guest) {
            Describe(out, return_address);
        }
        if (next <= fp) {
            break;
        }
        fp = next;
    }
    out.Str("\n");
#else
    (void)context;
#endif
    const int fd = open(g_path, O_WRONLY | O_APPEND | O_CLOEXEC);
    if (fd >= 0) {
        out.WriteTo(fd);
        fsync(fd);
        close(fd);
    }
}

void Handler(int sig, siginfo_t* info, void* raw_context) {
    bool report = g_reported == 0;
#ifdef __aarch64__
    if (report && (sig == SIGSEGV || sig == SIGBUS)) {
        const auto pc = static_cast<ucontext_t*>(raw_context)->uc_mcontext.pc;
        report = InGuestCode() || !IsManagedCode(pc);
    }
#endif
    if (report) {
        g_reported = 1;
        Report(sig, info, raw_context);
    }

    const struct sigaction& previous = g_previous[sig];
    if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
        previous.sa_sigaction(sig, info, raw_context);
    } else if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN &&
               previous.sa_handler != nullptr) {
        previous.sa_handler(sig);
    } else {
        // The default action: the faulting instruction runs again, or abort() raises again.
        struct sigaction default_action {};
        default_action.sa_handler = SIG_DFL;
        Common::SigAction(sig, &default_action, nullptr);
    }
}

} // namespace

void InstallCrashLog(const std::string& log_path) {
    if (log_path.size() >= sizeof(g_path)) {
        return;
    }
    std::memcpy(g_path, log_path.c_str(), log_path.size() + 1);
    for (const int sig : FatalSignals) {
        struct sigaction action {};
        action.sa_sigaction = Handler;
        action.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&action.sa_mask);
        Common::SigAction(sig, &action, &g_previous[sig]);
    }
}

} // namespace Common::Android
