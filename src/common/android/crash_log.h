// SPDX-FileCopyrightText: Copyright 2026 Lichii Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>

namespace Common::Android {

/// Appends a record of fatal native signals (signal, fault address, registers, the instruction at
/// the PC, whether it ran guest code, and a frame-pointer backtrace) to log_path before the
/// previous handler ends the process. The log's own buffered tail is lost on a crash, and Android
/// otherwise keeps the cause only in a tombstone the user cannot read. Install before NCE, which
/// passes host faults on to the handler it finds.
void InstallCrashLog(const std::string& log_path);

} // namespace Common::Android
