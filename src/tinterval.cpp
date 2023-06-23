#include <tinterval.h>

// Start time point.
tinterval tic_t()
{
	return std::chrono::steady_clock::now();
}

// End time point.
double toc_t(tinterval t0)
{
	return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t0).count();
}