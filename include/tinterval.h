// tic() and toc() operations giving time elapsed.
// 09/10/2022.
// zyao@ncut.edu.cn
#pragma once
#ifndef TINTERVAL_H
#define TINTERVAL_H

// Using chrono library.
#include <chrono>

typedef std::chrono::steady_clock::time_point tinterval;
// Alternatively, we can encapsulate this as a class or not.
tinterval tic_t();
double toc_t(tinterval t0);


#endif TINTERVAl_H
