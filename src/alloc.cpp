#include "alloc.hpp"
#include <stdexcept>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

void* AlignedAllocator::allocate(size_t size, size_t alignment) {
    if (size == 0) return nullptr;

    void* ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    if (!ptr) throw std::bad_alloc();
#else
    int ret = posix_memalign(&ptr, alignment, size);
    if (ret != 0) throw std::bad_alloc();
#endif
    return ptr;
}

void AlignedAllocator::deallocate(void* ptr) {
    if (!ptr) return;

#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

TensorPool::TensorPool(size_t initial_size) : total_allocated_(0), total_used_(0) {
    head_ = new Chunk{
        AlignedAllocator::allocate(initial_size),
        initial_size,
        nullptr
    };
    total_allocated_ = initial_size;
}

TensorPool::~TensorPool() {
    reset();

    Chunk* current = head_;
    while (current) {
        Chunk* next = current->next;
        AlignedAllocator::deallocate(current->data);
        delete current;
        current = next;
    }
}

void* TensorPool::allocate(size_t size) {
    // Simple first-fit allocation for now
    Chunk* current = head_;
    while (current) {
        if (current->size >= size && current->data) {
            void* result = current->data;
            if (current->size > size) {
                // Split chunk
                Chunk* remainder = new Chunk{
                    static_cast<char*>(current->data) + size,
                    current->size - size,
                    current->next
                };
                current->next = remainder;
            }
            current->data = nullptr;
            current->size = 0;
            total_used_ += size;
            return result;
        }
        current = current->next;
    }

    // Need to allocate new chunk
    size_t new_size = std::max(size, total_allocated_ / 2);
    Chunk* new_chunk = new Chunk{
        AlignedAllocator::allocate(new_size),
        new_size,
        head_
    };
    head_ = new_chunk;
    total_allocated_ += new_size;

    return allocate(size); // Recursive call will succeed
}

void TensorPool::reset() {
    Chunk* current = head_;
    while (current) {
        current->data = nullptr;
        current->size = 0;
        current = current->next;
    }
    total_used_ = 0;
}
