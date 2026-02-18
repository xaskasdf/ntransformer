#pragma once

#include <cstdio>
#include <cstdarg>

namespace nt {

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

inline LogLevel& global_log_level() {
    static LogLevel level = LogLevel::INFO;
    return level;
}

inline void set_log_level(LogLevel level) {
    global_log_level() = level;
}

inline void log(LogLevel level, const char* fmt, ...) {
    if (level < global_log_level()) return;

    const char* prefix = "";
    switch (level) {
        case LogLevel::DEBUG: prefix = "[DEBUG] "; break;
        case LogLevel::INFO:  prefix = "[INFO]  "; break;
        case LogLevel::WARN:  prefix = "[WARN]  "; break;
        case LogLevel::ERROR: prefix = "[ERROR] "; break;
    }

    fprintf(stderr, "%s", prefix);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

#define NT_LOG_DEBUG(...) ::nt::log(::nt::LogLevel::DEBUG, __VA_ARGS__)
#define NT_LOG_INFO(...)  ::nt::log(::nt::LogLevel::INFO,  __VA_ARGS__)
#define NT_LOG_WARN(...)  ::nt::log(::nt::LogLevel::WARN,  __VA_ARGS__)
#define NT_LOG_ERROR(...) ::nt::log(::nt::LogLevel::ERROR, __VA_ARGS__)

} // namespace nt
