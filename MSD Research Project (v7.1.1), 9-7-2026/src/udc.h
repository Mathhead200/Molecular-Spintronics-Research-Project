/**
 * @file udc.h
 * @author Christopher D'Angelo
 * @brief Defines the udc namespace.
 *        Contains globals and utilities used by other files.
 * 
 * @version 6.3
 * @date 2024-1-7
 * 
 * @copyright Copyright (c) 2011-2024
 */

#ifndef UDC_H
#define UDC_H

#include <cstring>
#include <exception>

/**
 * @brief Namesapce for MSD class, and all related identifiers.
 *        Named for the University of the District of Columbia where
 *        the Center for Nanotechnology Research and Education is established.
 */
namespace udc {

using std::exception;


/** Mathematical constant: Euler's Number */
const double E = 2.71828182845904523536;

/** Mathematical constant */
const double PI = 3.14159265358979323846;


/**
 * @brief Square a number, fast.
 * 
 * @param x The base of the exponential expression x^2 (x raised to the 2nd power).
 * @return Result after evaluating the exponential expression.
 */
inline double sq(double x) {
	return x * x;
}

/**
 * @brief Cube a number, fast.
 * 
 * @param x The base of the exponential expression x^3 (x raised to the 3rd power).
 * @return Result after evaluating the exponential expression.
 */
inline double cube(double x) {
	return x * x * x;
}

/**
 * Buffer/Binary Read:
 * 
 * Reads the given number of bytes from the buffer and stores it into the destination,
 * advancing the buffer pointer in the process. The pointers should not overlap!
 * 
 * @param destination Where data will be stored
 * @param dSize Amount of data to copy (in bytes).
 * @param buffer Source of the data. This pointer will be advance by dSize.
 */
inline void bread(void * destination, size_t dSize, const unsigned char * &buffer) {
	memcpy(destination, buffer, dSize);
	buffer += dSize;
}

/**
 * Templated version: determines the approprate number of bytes by calling "sizeof(T)".
 * @see udc#bread(void *, size_t, const unsigned char *&)
 */
template <typename T> inline void bread(T &destination, const unsigned char * &buffer) {
	bread(&destination, sizeof(T), buffer);
}

/**
 * Buffer/Binary Write:
 * 
 * Writes the given number of bytes from the source and stores it into the buffer,
 * advancing the buffer pointer in the process. The pointers should not overlap!
 * 
 * @param source Data to be copied.
 * @param sSize Amount of data to copy (in bytes).
 * @param buffer Destination, where the data. This pointer will be advance by dSize.
 */
inline void bwrite(const void * source, size_t sSize, unsigned char * &buffer) {
	memcpy(buffer, source, sSize);
	buffer += sSize;
}

/**
 * Templated version: determines the approprate number of bytes by calling "sizeof(T)".
 * @see udc#bwrite(const void *, size_t, unsigned char *&)
 */
template <typename T> inline void bwrite(const T &source, unsigned char * &buffer) {
	bwrite(&source, sizeof(T), buffer);
}


/** @brief Custom exception type. */
class UDCException : public exception {
 private:
	const char *message;

 public:
	/** @param message A message detailing the exception. */
	UDCException(const char *message) : message(message) {}
	
	/** @see std::exception#what() const */
	virtual const char * what() const noexcept { return message; }
};


} //end of namespace

#endif