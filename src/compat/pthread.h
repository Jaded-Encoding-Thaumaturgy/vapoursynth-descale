#ifndef PTHREAD_COMPAT_H
#define PTHREAD_COMPAT_H

// Minimal pthread-compat shim for Windows/MSVC.

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef CRITICAL_SECTION pthread_mutex_t;
typedef void* pthread_mutexattr_t;

static inline int pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t *attr) {
    (void)attr;
    InitializeCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t *mutex) {
    EnterCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *mutex) {
    LeaveCriticalSection(mutex);
    return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t *mutex) {
    DeleteCriticalSection(mutex);
    return 0;
}

#endif
#endif
