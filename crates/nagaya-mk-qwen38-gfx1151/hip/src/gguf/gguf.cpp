// SPDX-License-Identifier: MIT
#include "gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "dequant.hpp"

namespace mkgguf {

namespace {

constexpr uint32_t kGgufMagic = 0x46554747u;  // ASCII "GGUF" read as a little-endian uint32
constexpr uint32_t kGgufVersion = 3;
constexpr uint32_t kDefaultAlignment = 32;
constexpr size_t kMaxStringLen = 1u << 30;  // sanity cap, mirrors gguf-py's GGUF_MAX_STRING_LENGTH order

// A bounds-checked cursor over one shard's mmap'd bytes. Every read that would run past the
// end of the file throws, with the shard path and offset, instead of reading whatever memory
// happens to follow the mapping.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size, std::string shard_path)
        : base_(base), size_(size), shard_path_(std::move(shard_path)) {}

    template <typename T>
    T get(size_t& pos) {
        static_assert(std::is_trivially_copyable_v<T>);
        require(pos, sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos, sizeof(T));
        pos += sizeof(T);
        return v;
    }

    std::string get_str(size_t& pos) {
        uint64_t len = get<uint64_t>(pos);
        if (len > kMaxStringLen) {
            fail(pos, "string length " + std::to_string(len) + " exceeds the sanity cap");
        }
        require(pos, len);
        std::string s(reinterpret_cast<const char*>(base_ + pos), static_cast<size_t>(len));
        pos += len;
        return s;
    }

    void require(size_t pos, uint64_t n) const {
        // pos + n can overflow size_t on a crafted file; check both directions.
        if (n > size_ || pos > size_ - n) {
            fail(pos, "read of " + std::to_string(n) + " bytes runs past the shard's " +
                          std::to_string(size_) + " bytes");
        }
    }

    [[noreturn]] void fail(size_t pos, const std::string& msg) const {
        throw std::runtime_error("mkgguf: " + shard_path_ + " @" + std::to_string(pos) + ": " + msg);
    }

    const uint8_t* base_;
    size_t size_;
    std::string shard_path_;
};

KV parse_kv_value(Cursor& c, size_t& pos, GgufValueType type) {
    KV v;
    v.type = type;
    switch (type) {
        case GgufValueType::UINT8: v.i = c.get<uint8_t>(pos); return v;
        case GgufValueType::INT8: v.i = c.get<int8_t>(pos); return v;
        case GgufValueType::UINT16: v.i = c.get<uint16_t>(pos); return v;
        case GgufValueType::INT16: v.i = c.get<int16_t>(pos); return v;
        case GgufValueType::UINT32: v.i = c.get<uint32_t>(pos); return v;
        case GgufValueType::INT32: v.i = c.get<int32_t>(pos); return v;
        case GgufValueType::UINT64: v.i = static_cast<int64_t>(c.get<uint64_t>(pos)); return v;
        case GgufValueType::INT64: v.i = c.get<int64_t>(pos); return v;
        case GgufValueType::FLOAT32: v.f = c.get<float>(pos); return v;
        case GgufValueType::FLOAT64: v.f = c.get<double>(pos); return v;
        case GgufValueType::BOOL: v.b = c.get<uint8_t>(pos) != 0; return v;
        case GgufValueType::STRING: v.s = c.get_str(pos); return v;
        case GgufValueType::ARRAY: {
            uint32_t elem_type_raw = c.get<uint32_t>(pos);
            auto elem_type = static_cast<GgufValueType>(elem_type_raw);
            if (elem_type == GgufValueType::ARRAY) {
                c.fail(pos, "nested arrays are not supported");
            }
            uint64_t len = c.get<uint64_t>(pos);
            v.elem_type = elem_type;
            v.arr_i.reserve(static_cast<size_t>(len));
            for (uint64_t idx = 0; idx < len; ++idx) {
                KV item = parse_kv_value(c, pos, elem_type);
                switch (elem_type) {
                    case GgufValueType::STRING:
                        v.arr_s.push_back(std::move(item.s));
                        break;
                    case GgufValueType::FLOAT32:
                    case GgufValueType::FLOAT64:
                        v.arr_f.push_back(item.f);
                        break;
                    case GgufValueType::BOOL:
                        v.arr_i.push_back(item.b ? 1 : 0);
                        break;
                    default:
                        v.arr_i.push_back(item.i);
                        break;
                }
            }
            return v;
        }
    }
    c.fail(pos, "unknown GGUF value type id " + std::to_string(static_cast<uint32_t>(type)));
}

std::string to_hex(uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", v);
    return buf;
}

}  // namespace

Model::Model(const std::vector<std::string>& shard_paths) {
    if (shard_paths.empty()) {
        throw std::runtime_error("mkgguf: Model: no shard paths given");
    }
    shards_.resize(shard_paths.size());
    for (size_t i = 0; i < shard_paths.size(); ++i) {
        parse_shard(i, shard_paths);
    }
}

Model::~Model() {
    for (auto& sh : shards_) {
        if (sh.map != nullptr && sh.map != MAP_FAILED) {
            ::munmap(sh.map, sh.size);
        }
        if (sh.fd >= 0) {
            ::close(sh.fd);
        }
    }
}

void Model::parse_shard(size_t shard_idx, const std::vector<std::string>& shard_paths) {
    const std::string& path = shard_paths[shard_idx];
    Shard& sh = shards_[shard_idx];
    sh.path = path;

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("mkgguf: open " + path + ": " + std::strerror(errno));
    }
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error("mkgguf: fstat " + path + ": " + std::strerror(e));
    }
    size_t size = static_cast<size_t>(st.st_size);
    if (size < 24) {
        ::close(fd);
        throw std::runtime_error("mkgguf: " + path + ": file of " + std::to_string(size) +
                                  " bytes is too small to hold a GGUF header");
    }
    // Read-only mmap of the whole shard: this is the only way an 88 GB model gets read on a
    // box with far less than 88 GB of headroom. Nothing here copies tensor data into
    // anonymous memory.
    void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        int e = errno;
        ::close(fd);
        throw std::runtime_error("mkgguf: mmap " + path + ": " + std::strerror(e));
    }
    sh.fd = fd;
    sh.map = map;
    sh.size = size;

    const uint8_t* base = static_cast<const uint8_t*>(map);
    Cursor c(base, size, path);
    size_t pos = 0;

    uint32_t magic = c.get<uint32_t>(pos);
    if (magic != kGgufMagic) {
        c.fail(pos, "bad GGUF magic " + to_hex(magic) + " (expected " + to_hex(kGgufMagic) + ")");
    }
    uint32_t version = c.get<uint32_t>(pos);
    if (version != kGgufVersion) {
        c.fail(pos, "GGUF version " + std::to_string(version) +
                        " != 3; this reader only speaks GGUF v3");
    }
    uint64_t tensor_count = c.get<uint64_t>(pos);
    uint64_t kv_count = c.get<uint64_t>(pos);

    std::unordered_map<std::string, KV> shard_kv;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.get_str(pos);
        uint32_t raw_type = c.get<uint32_t>(pos);
        KV value = parse_kv_value(c, pos, static_cast<GgufValueType>(raw_type));
        if (!shard_kv.emplace(key, std::move(value)).second) {
            c.fail(pos, "duplicate KV key '" + key + "' within one shard");
        }
    }

    // Alignment: default 32, overridden by this shard's own general.alignment (u32). Each
    // shard is a fully valid standalone GGUF file, so this must be resolved per-shard.
    uint32_t alignment = kDefaultAlignment;
    if (auto it = shard_kv.find("general.alignment"); it != shard_kv.end()) {
        if (it->second.type != GgufValueType::UINT32) {
            c.fail(pos, "general.alignment has the wrong KV type");
        }
        int64_t a = it->second.i;
        if (a <= 0 || (a & (a - 1)) != 0) {
            c.fail(pos, "general.alignment " + std::to_string(a) + " is not a power of two");
        }
        alignment = static_cast<uint32_t>(a);
    }

    struct RawTensorInfo {
        std::string name;
        uint32_t n_dims = 0;
        uint64_t ne[4] = {1, 1, 1, 1};
        uint32_t type = 0;
        uint64_t offset = 0;
    };
    std::vector<RawTensorInfo> raw_infos;
    raw_infos.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        RawTensorInfo ri;
        ri.name = c.get_str(pos);
        ri.n_dims = c.get<uint32_t>(pos);
        if (ri.n_dims == 0 || ri.n_dims > 4) {
            c.fail(pos, "tensor '" + ri.name + "' has n_dims=" + std::to_string(ri.n_dims) +
                            " (must be 1..4)");
        }
        for (uint32_t d = 0; d < ri.n_dims; ++d) {
            ri.ne[d] = c.get<uint64_t>(pos);
        }
        ri.type = c.get<uint32_t>(pos);
        ri.offset = c.get<uint64_t>(pos);
        raw_infos.push_back(std::move(ri));
    }

    size_t data_offset = pos;
    size_t rem = data_offset % alignment;
    if (rem != 0) {
        data_offset += (alignment - rem);
    }
    sh.data_offset = data_offset;

    // This model is a split GGUF: every shard must declare the same split.count (equal to how
    // many shard paths we were given) and a split.no matching this shard's position in that
    // list. That's the caller's promise that "these paths, in this order" really is one split
    // set, checked rather than assumed.
    auto require_shard_i64 = [&](const char* key) -> int64_t {
        auto it = shard_kv.find(key);
        if (it == shard_kv.end()) {
            c.fail(pos, std::string("missing required split key '") + key + "'");
        }
        return it->second.i;
    };
    int64_t split_count = require_shard_i64("split.count");
    int64_t split_no = require_shard_i64("split.no");
    if (split_count < 0 || static_cast<size_t>(split_count) != shard_paths.size()) {
        c.fail(pos, "split.count=" + std::to_string(split_count) + " but " +
                        std::to_string(shard_paths.size()) + " shard paths were given");
    }
    if (split_no < 0 || static_cast<size_t>(split_no) != shard_idx) {
        c.fail(pos, "split.no=" + std::to_string(split_no) + " but this shard is at position " +
                        std::to_string(shard_idx) +
                        " in the given list; pass shard paths in split order");
    }

    for (auto& ri : raw_infos) {
        if (!ggml_type_is_known(ri.type)) {
            c.fail(pos, "tensor '" + ri.name + "' has unknown ggml type id " +
                            std::to_string(ri.type));
        }
        GgmlType gt = static_cast<GgmlType>(ri.type);
        GgmlTypeTraits traits = ggml_type_traits(gt);

        uint64_t n_elements = 1;
        for (uint32_t d = 0; d < 4; ++d) {
            n_elements *= ri.ne[d];
        }
        if (n_elements % traits.block_size != 0) {
            c.fail(pos, "tensor '" + ri.name + "': n_elements " + std::to_string(n_elements) +
                            " is not a multiple of " + ggml_type_name(gt) + "'s block size " +
                            std::to_string(traits.block_size));
        }
        uint64_t n_bytes = n_elements / traits.block_size * static_cast<uint64_t>(traits.type_size);

        if (ri.offset % alignment != 0) {
            c.fail(pos, "tensor '" + ri.name + "' offset " + std::to_string(ri.offset) +
                            " is not " + std::to_string(alignment) + "-aligned");
        }
        uint64_t abs_start = static_cast<uint64_t>(data_offset) + ri.offset;
        if (abs_start > size || n_bytes > size - abs_start) {
            c.fail(pos, "tensor '" + ri.name + "' data [" + std::to_string(abs_start) + ", " +
                            std::to_string(abs_start + n_bytes) + ") runs past the shard's " +
                            std::to_string(size) + " bytes");
        }

        TensorInfo info;
        info.name = ri.name;
        info.type = gt;
        info.n_dims = ri.n_dims;
        for (int d = 0; d < 4; ++d) {
            info.ne[d] = ri.ne[d];
        }
        info.n_elements = n_elements;
        info.n_bytes = n_bytes;
        info.data = base + abs_start;

        if (!tensor_index_.emplace(info.name, tensors_.size()).second) {
            c.fail(pos, "duplicate tensor name '" + info.name + "' across shards");
        }
        tensors_.push_back(std::move(info));
    }

    for (auto& [key, value] : shard_kv) {
        merge_kv(key, std::move(value), path);
    }
}

void Model::merge_kv(const std::string& key, KV&& value, const std::string& shard_path) {
    // split.* is per-shard bookkeeping (already validated in parse_shard) and legitimately
    // differs shard to shard (split.no), so it never joins the merged, model-wide KV store.
    if (key.rfind("split.", 0) == 0) {
        return;
    }
    if (kv_.find(key) != kv_.end()) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is duplicated across shards (" +
                                  shard_path + " and an earlier one); expected each non-split "
                                  "key to live in exactly one shard");
    }
    kv_.emplace(key, std::move(value));
}

const TensorInfo& Model::tensor(size_t index) const {
    if (index >= tensors_.size()) {
        throw std::runtime_error("mkgguf: tensor index " + std::to_string(index) +
                                  " out of range (n_tensors=" + std::to_string(tensors_.size()) + ")");
    }
    return tensors_[index];
}

const TensorInfo& Model::tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
        throw std::runtime_error("mkgguf: no tensor named '" + name + "'");
    }
    return tensors_[it->second];
}

bool Model::has_tensor(const std::string& name) const {
    return tensor_index_.find(name) != tensor_index_.end();
}

bool Model::has_kv(const std::string& key) const {
    return kv_.find(key) != kv_.end();
}

const KV& Model::require_kv(const std::string& key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) {
        throw std::runtime_error("mkgguf: no KV metadata key '" + key + "'");
    }
    return it->second;
}

int64_t Model::get_i64(const std::string& key) const {
    const KV& v = require_kv(key);
    switch (v.type) {
        case GgufValueType::UINT8:
        case GgufValueType::INT8:
        case GgufValueType::UINT16:
        case GgufValueType::INT16:
        case GgufValueType::UINT32:
        case GgufValueType::INT32:
        case GgufValueType::UINT64:
        case GgufValueType::INT64:
            return v.i;
        case GgufValueType::BOOL:
            return v.b ? 1 : 0;
        default:
            throw std::runtime_error("mkgguf: KV key '" + key + "' is not an integer (GGUF type " +
                                      std::to_string(static_cast<uint32_t>(v.type)) + ")");
    }
}

double Model::get_f64(const std::string& key) const {
    const KV& v = require_kv(key);
    switch (v.type) {
        case GgufValueType::FLOAT32:
        case GgufValueType::FLOAT64:
            return v.f;
        case GgufValueType::UINT8:
        case GgufValueType::INT8:
        case GgufValueType::UINT16:
        case GgufValueType::INT16:
        case GgufValueType::UINT32:
        case GgufValueType::INT32:
        case GgufValueType::UINT64:
        case GgufValueType::INT64:
            return static_cast<double>(v.i);
        default:
            throw std::runtime_error("mkgguf: KV key '" + key + "' is not a float (GGUF type " +
                                      std::to_string(static_cast<uint32_t>(v.type)) + ")");
    }
}

bool Model::get_bool(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::BOOL) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not a bool (GGUF type " +
                                  std::to_string(static_cast<uint32_t>(v.type)) + ")");
    }
    return v.b;
}

const std::string& Model::get_str(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::STRING) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not a string (GGUF type " +
                                  std::to_string(static_cast<uint32_t>(v.type)) + ")");
    }
    return v.s;
}

size_t Model::get_arr_len(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::ARRAY) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not an array");
    }
    switch (v.elem_type) {
        case GgufValueType::STRING: return v.arr_s.size();
        case GgufValueType::FLOAT32:
        case GgufValueType::FLOAT64: return v.arr_f.size();
        default: return v.arr_i.size();
    }
}

std::vector<int64_t> Model::get_arr_i64(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::ARRAY) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not an array");
    }
    switch (v.elem_type) {
        case GgufValueType::UINT8: case GgufValueType::INT8: case GgufValueType::UINT16:
        case GgufValueType::INT16: case GgufValueType::UINT32: case GgufValueType::INT32:
        case GgufValueType::UINT64: case GgufValueType::INT64: case GgufValueType::BOOL:
            return v.arr_i;
        default:
            throw std::runtime_error("mkgguf: KV array '" + key + "' elements are not integers");
    }
}

std::vector<double> Model::get_arr_f64(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::ARRAY) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not an array");
    }
    if (v.elem_type == GgufValueType::FLOAT32 || v.elem_type == GgufValueType::FLOAT64) {
        return v.arr_f;
    }
    throw std::runtime_error("mkgguf: KV array '" + key + "' elements are not floats");
}

uint64_t Model::data_offset(size_t shard) const {
    if (shard >= shards_.size()) {
        throw std::runtime_error("mkgguf: shard " + std::to_string(shard) + " of " + std::to_string(shards_.size()));
    }
    return shards_[shard].data_offset;
}

std::vector<std::string> Model::get_arr_str(const std::string& key) const {
    const KV& v = require_kv(key);
    if (v.type != GgufValueType::ARRAY || v.elem_type != GgufValueType::STRING) {
        throw std::runtime_error("mkgguf: KV key '" + key + "' is not a string array");
    }
    return v.arr_s;
}

void Model::dequant_rows(const std::string& name, uint64_t row0, uint64_t row1, float* out) const {
    const TensorInfo& t = tensor(name);
    GgmlTypeTraits traits = ggml_type_traits(t.type);
    uint64_t ne0 = t.ne[0];
    if (ne0 % traits.block_size != 0) {
        throw std::runtime_error("mkgguf: tensor '" + name + "': ne[0]=" + std::to_string(ne0) +
                                  " is not a multiple of " + ggml_type_name(t.type) +
                                  "'s block size " + std::to_string(traits.block_size));
    }
    uint64_t n_rows = t.n_elements / ne0;
    if (row1 < row0 || row1 > n_rows) {
        throw std::runtime_error("mkgguf: tensor '" + name + "': row range [" +
                                  std::to_string(row0) + ", " + std::to_string(row1) +
                                  ") out of bounds (n_rows=" + std::to_string(n_rows) + ")");
    }
    uint64_t row_bytes = ne0 / traits.block_size * static_cast<uint64_t>(traits.type_size);
    const uint8_t* src = t.data + row0 * row_bytes;
    for (uint64_t r = row0; r < row1; ++r) {
        dequant_f32(t.type, src, static_cast<int64_t>(ne0), out);
        src += row_bytes;
        out += ne0;
    }
}

}  // namespace mkgguf

namespace mkgguf {

void Model::drop_cached(const uint8_t* p, uint64_t n) const {
    const Shard* sh = nullptr;
    for (const auto& s : shards_) {
        const uint8_t* base = static_cast<const uint8_t*>(s.map);
        if (p >= base && p < base + s.size) {
            sh = &s;
            break;
        }
    }
    if (sh == nullptr) {
        throw std::runtime_error("mkgguf: drop_cached: range at " + std::to_string(uintptr_t(p)) +
                                  " is not inside any shard's mapping");
    }
    const uint8_t* base = static_cast<const uint8_t*>(sh->map);
    const uint64_t lo = static_cast<uint64_t>(p - base);
    if (n > sh->size - lo) {
        throw std::runtime_error("mkgguf: drop_cached: range [" + std::to_string(lo) + ", +" +
                                  std::to_string(n) + ") runs past " + sh->path + "'s " +
                                  std::to_string(sh->size) + " bytes");
    }
    // Inward to whole pages: a page shared with bytes outside the range stays.
    const uint64_t pg = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
    const uint64_t first = (lo + pg - 1) / pg * pg, end = (lo + n) / pg * pg;
    if (end <= first) return;
    if (::madvise(static_cast<uint8_t*>(sh->map) + first, end - first, MADV_DONTNEED) != 0) {
        throw std::runtime_error("mkgguf: drop_cached: madvise " + sh->path + ": " + std::strerror(errno));
    }
    int e = ::posix_fadvise(sh->fd, static_cast<off_t>(first), static_cast<off_t>(end - first),
                            POSIX_FADV_DONTNEED);
    if (e != 0) {
        throw std::runtime_error("mkgguf: drop_cached: posix_fadvise " + sh->path + ": " + std::strerror(e));
    }
}

}  // namespace mkgguf
