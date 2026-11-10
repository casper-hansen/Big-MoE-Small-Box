// penv("NAME") reads the environment variable <prefix>NAME. Each backend sets its prefix in BIGMOE_ENGINE_PREFIX
// (GLM-5.3: GLM53_, DeepSeek-V4.1: DSV41_; unset: BIGMOE_), so every backend has its own variable names.
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
