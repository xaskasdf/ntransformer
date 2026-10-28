#pragma once
#include <cstdlib>
void gap_trace_begin(const char* name);
void gap_trace_end();
inline bool gap_tracing(int round) { return round == 2 && std::getenv("GAP_TRACE"); }
