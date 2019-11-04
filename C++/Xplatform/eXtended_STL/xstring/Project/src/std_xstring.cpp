#include "std_xstring.h"

#if defined(__unix__)
template <class _Kty>
_NODISCARD size_t std::hash_xstring_vals::_Hash_array_representation(
    const _Kty* const _First, const size_t _Count) const noexcept {
    // bitwise hashes the representation of an array
    static_assert(is_trivial_v<_Kty>, "Only trivial types can be directly hashed.");
    return _Fnv1a_append_bytes(
        _FNV_offset_basis, reinterpret_cast<const unsigned char*>(_First), _Count * sizeof(_Kty)
    );
}

_NODISCARD inline size_t std::hash_xstring_vals::_Fnv1a_append_bytes(
    size_t _Val, const unsigned char* const _First, const size_t _Count) const noexcept {
   // accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
    for (size_t _Idx = 0; _Idx < _Count; ++_Idx) {
        _Val ^= static_cast<size_t>(_First[_Idx]);
        _Val *= _FNV_prime;
    }
    return _Val;
}
#endif
// STRUCT TEMPLATE SPECIALIZATION hash
size_t std::hash<xstring>::operator()(const xstring& _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval.c_str(), _Keyval.size());
};

size_t std::hash<xstring*>::operator()(const xstring* _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval->c_str(), _Keyval->size());
};

size_t std::hash<const xstring*>::operator()(const xstring* _Keyval) const noexcept {
    return _Hash_array_representation(_Keyval->c_str(), _Keyval->size());
};
