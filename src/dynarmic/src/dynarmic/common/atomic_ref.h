// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

namespace Dynarmic::Common {

#ifdef __cpp_lib_atomic_ref
template<typename T>
using AtomicRef = std::atomic_ref<T>;
#else
// The Android NDK's libc++ has no std::atomic_ref yet; this covers what the exclusive monitor uses.
template<typename T>
class AtomicRef {
public:
    explicit AtomicRef(T& obj) noexcept
            : ptr{&obj} {}

    T load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return __atomic_load_n(ptr, ToBuiltin(order));
    }

    void store(T value, std::memory_order order = std::memory_order_seq_cst) const noexcept {
        __atomic_store_n(ptr, value, ToBuiltin(order));
    }

    bool compare_exchange_strong(T& expected, T desired,
                                 std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return __atomic_compare_exchange_n(ptr, &expected, desired, false, ToBuiltin(order),
                                           FailureOrder(order));
    }

private:
    static constexpr int ToBuiltin(std::memory_order order) {
        switch (order) {
        case std::memory_order_relaxed:
            return __ATOMIC_RELAXED;
        case std::memory_order_consume:
            return __ATOMIC_CONSUME;
        case std::memory_order_acquire:
            return __ATOMIC_ACQUIRE;
        case std::memory_order_release:
            return __ATOMIC_RELEASE;
        case std::memory_order_acq_rel:
            return __ATOMIC_ACQ_REL;
        default:
            return __ATOMIC_SEQ_CST;
        }
    }

    // A failed compare-exchange only loads, so it cannot use release semantics.
    static constexpr int FailureOrder(std::memory_order order) {
        switch (order) {
        case std::memory_order_release:
            return __ATOMIC_RELAXED;
        case std::memory_order_acq_rel:
            return __ATOMIC_ACQUIRE;
        default:
            return ToBuiltin(order);
        }
    }

    T* ptr;
};
#endif

}  // namespace Dynarmic::Common
