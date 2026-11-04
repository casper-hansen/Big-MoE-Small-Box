// Per-model environment prefix of the shared offload engine: knob NAME is read from <BIGMOE_ENGINE_PREFIX>NAME
// (GLM-5.3: GLM53_, DeepSeek-V4.1: DSV41_), so every backend keeps its own variable names.
#pragma once
#include <cstdlib>
#include <string>

inline const std::string& nv2_env_prefix()
{
    static const std::string p = [] {
        const char* v = std::getenv("BIGMOE_ENGINE_PREFIX");
        return std::string(v && *v ? v : "BIGMOE_");
    }();
    return p;
}

inline const char* penv(const char* name)
{
    const std::string k = nv2_env_prefix() + name;
    return std::getenv(k.c_str());
}
