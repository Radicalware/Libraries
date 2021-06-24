#pragma once
#include "xstring.h"

#if defined(__unix__)
#define   _NODISCARD
#define   _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS
constexpr size_t _FNV_offset_basis = 14695981039346656037ULL;
constexpr size_t _FNV_prime = 1099511628211ULL;
#endif

namespace std
{
    struct hash_xstring_vals
    {
        _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef xstring argument_type;
        _CXX17_DEPRECATE_ADAPTOR_TYPEDEFS typedef size_t result_type;

#if defined(__unix__) // -----------------------------------------------------------------------------
        template <class _Kty>
        _NODISCARD size_t _Hash_array_representation(
            const _Kty* const _First, const size_t _Count) const noexcept;

        _NODISCARD inline size_t _Fnv1a_append_bytes(
            size_t _Val, const unsigned char* const _First, const size_t _Count) const noexcept;
#endif // --------------------------------------------------------------------------------------------
    };

    // STRUCT TEMPLATE SPECIALIZATION hash
    template<> struct hash<xstring> : hash_xstring_vals {
        _NODISCARD size_t operator()(const xstring& _Keyval) const noexcept;
    };

    template<> struct hash<xstring*> : hash_xstring_vals {
        _NODISCARD size_t operator()(const xstring* _Keyval) const noexcept;
    };

    template<> struct hash<const xstring*> : hash_xstring_vals {
        _NODISCARD size_t operator()(const xstring* _Keyval) const noexcept;
    };
}
