#ifndef ALLOC_HPP
#define ALLOC_HPP

#include <memory>
#include <cstddef>
#include <cstdlib>

class AlignedAllocator {
public:
    static void* allocate(size_t size, size_t alignment = 32);
    static void deallocate(void* ptr);

    template<typename T>
    static std::unique_ptr<T[], decltype(&deallocate)> make_unique_aligned(size_t count) {
        auto* ptr = static_cast<T*>(allocate(count * sizeof(T)));
        return std::unique_ptr<T[], decltype(&deallocate)>(ptr, &deallocate);
    }
};

// Memory pool for frequently allocated tensors
class TensorPool {
public:
    TensorPool(size_t initial_size = 1024 * 1024); // 1MB default
    ~TensorPool();

    void* allocate(size_t size);
    void reset();

private:
    struct Chunk {
        void* data;
        size_t size;
        Chunk* next;
    };

    Chunk* head_;
    size_t total_allocated_;
    size_t total_used_;
};

#endif // ALLOC_HPP
