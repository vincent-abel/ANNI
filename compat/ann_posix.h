/*
**	Thin POSIX shims for native Windows builds.
**	On Linux/macOS this header is a no-op include guard — keep existing
**	#include <unistd.h> / <sys/time.h> / <pthread.h> paths working.
*/
#ifndef ANN_POSIX_H__
#define ANN_POSIX_H__

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <direct.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdint.h>

#ifndef R_OK
#define R_OK 4
#endif
#ifndef W_OK
#define W_OK 2
#endif
#ifndef X_OK
#define X_OK 1
#endif
#ifndef F_OK
#define F_OK 0
#endif

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

/* access(2) / isatty(3) */
#define access _access
#define isatty _isatty
#define open _open
#define close _close
#define read _read
#define write _write
#define fileno _fileno

/* --- gettimeofday / nanosleep --- */
struct timezone {
	int tz_minuteswest;
	int tz_dsttime;
};

struct timeval {
	long tv_sec;
	long tv_usec;
};

static inline int gettimeofday(struct timeval *tv, struct timezone *tz)
{
	FILETIME ft;
	ULARGE_INTEGER uli;
	unsigned long long us;

	(void)tz;
	if (!tv)
		return -1;
	GetSystemTimeAsFileTime(&ft);
	uli.LowPart = ft.dwLowDateTime;
	uli.HighPart = ft.dwHighDateTime;
	/* FILETIME is 100ns since 1601-01-01; Unix epoch offset: */
	us = (uli.QuadPart / 10ULL) - 11644473600000000ULL;
	tv->tv_sec = (long)(us / 1000000ULL);
	tv->tv_usec = (long)(us % 1000000ULL);
	return 0;
}

static inline int nanosleep(const struct timespec *req, struct timespec *rem)
{
	DWORD ms;

	(void)rem;
	if (!req)
		return -1;
	ms = (DWORD)(req->tv_sec * 1000L + req->tv_nsec / 1000000L);
	if (ms == 0 && (req->tv_sec > 0 || req->tv_nsec > 0))
		ms = 1;
	Sleep(ms);
	return 0;
}

/* --- pthread mutex via SRWLOCK (static-init friendly) --- */
typedef SRWLOCK pthread_mutex_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT

static inline int pthread_mutex_lock(pthread_mutex_t *m)
{
	AcquireSRWLockExclusive(m);
	return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t *m)
{
	ReleaseSRWLockExclusive(m);
	return 0;
}

/* mmap: not provided — callers should skip under _WIN32 (fread path). */
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)(intptr_t)-1)
#endif

#else /* !_WIN32 */

/* Linux: nothing — use system headers from the .c files as before. */

#endif /* _WIN32 */

#endif /* ANN_POSIX_H__ */
