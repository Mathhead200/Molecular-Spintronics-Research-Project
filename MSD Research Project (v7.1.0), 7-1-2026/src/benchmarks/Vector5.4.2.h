/*
 * Vector.h
 *
 *  Last Edited: Aug 27, 2013
 *       Author: Christopher D'Angelo
 */

#ifndef UDC_VECTOR_5_4_2
#define UDC_VECTOR_5_4_2

#include <cmath>
#include <iostream>
#include "../udc.h"


namespace udc_5_4_2 {

using std::ostream;

using udc::PI;


class Vector {
 private:
	static double turn(double angle); // Rotate the angle PI radians; stays in [PI, -PI).

 public:
	double x;
	double y;
	double z;

	Vector(double x, double y, double z);
	Vector(double x, double y);
	Vector();

	static Vector ZERO;
	static Vector I;
	static Vector J;
	static Vector K;

	static Vector cylindricalForm(double r, double theta, double z);
	static Vector polarForm(double r, double theta);
	static Vector sphericalForm(double rho, double theta, double phi);

	double norm() const;
	double theta() const;
	double phi() const;

	bool operator ==(const Vector &) const;
	bool operator !=(const Vector &) const;

	Vector operator +(const Vector &) const;
	Vector operator -() const;
	Vector operator -(const Vector &) const;
	Vector operator *(double) const;
	double operator *(const Vector &) const; //dot product

	double distanceSq(const Vector &) const;
	double distance(const Vector &) const;
	double dotProduct(const Vector &) const;
	double angleBetween(const Vector &) const;
	Vector crossProduct(Vector v) const;

	Vector& operator +=(const Vector &);
	Vector& operator -=(const Vector &);
	Vector& operator *=(double);

	Vector& negate();
	Vector& rotate(double theta, double phi);
	Vector& rotate(double theta);
	Vector& normalize();
};


//--------------------------------------------------------------------------------
double Vector::turn(double angle) {
	return angle + (angle <= 0 ? PI : -PI);
}


//--------------------------------------------------------------------------------
Vector Vector::ZERO(0, 0, 0);
Vector Vector::I(1, 0, 0);
Vector Vector::J(0, 1, 0);
Vector Vector::K(0, 0, 1);


//--------------------------------------------------------------------------------
Vector::Vector(double x, double y, double z) : x(x), y(y), z(z) {
}

Vector::Vector(double x, double y) : x(x), y(y), z(0) {
}

Vector::Vector() : x(0), y(0), z(0) {
}


//-------------------------------------------------------------------------------
Vector Vector::cylindricalForm(double r, double theta, double z) {
	return Vector( r * cos(theta),  r * sin(theta), z );
}

Vector Vector::polarForm(double r, double theta) {
	return cylindricalForm(r, theta, 0);
}

Vector Vector::sphericalForm(double rho, double theta, double phi) {
	return cylindricalForm( rho * cos(phi), theta, rho * sin(phi) );
}


//-------------------------------------------------------------------------------
double Vector::norm() const {
	return sqrt( x * x + y * y + z * z );
}

double Vector::theta() const {
	return atan2(y, x);
}

double Vector::phi() const {
	double r = sqrt( x * x + y * y );
	if( r == 0 ) {
		if( z > 0 )
			return PI / 2;
		else if( z < 0 )
			return -PI / 2;
		else
			return 0;
	} else
		return atan( z / r );
}


bool Vector::operator ==(const Vector &v) const {
	return x == v.x && y == v.y && z == v.z;
}

bool Vector::operator !=(const Vector &v) const {
	return x != v.x || y != v.y || z != v.z;
}


Vector Vector::operator +(const Vector &v) const {
	return Vector(x + v.x, y + v.y, z + v.z);
}

Vector Vector::operator -() const {
	return Vector(-x, -y, -z);
}

Vector Vector::operator -(const Vector &v) const {
	return Vector(x - v.x, y - v.y, z - v.z);
}

Vector Vector::operator *(double k) const {
	return Vector(k * x, k * y, k * z);
}

double Vector::operator *(const Vector &v) const {
	return dotProduct(v);
}


double Vector::distanceSq(const Vector &v) const {
	double dx = v.x - x, dy = v.y - y, dz = v.z - z;
	return dx * dx + dy * dy + dz * dz;
}

double Vector::distance(const Vector &v) const {
	return sqrt( distanceSq(v) );
}

double Vector::dotProduct(const Vector &v) const {
	return x * v.x + y * v.y + z * v.z;
}

double Vector::angleBetween(const Vector &v) const {
	return acos( dotProduct(v) / (norm() * v.norm()) );
}

Vector Vector::crossProduct(Vector v) const {
	return Vector( y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x );
}


Vector& Vector::operator +=(const Vector &v) {
	return *this = *this + v;
}

Vector& Vector::operator -=(const Vector &v) {
	return *this = *this - v;
}

Vector& Vector::operator *=(double k) {
	return *this = *this * k;
}


Vector& Vector::negate() {
	return *this = -*this;
}

Vector& Vector::rotate(double theta, double phi) {
	return *this = sphericalForm( norm(), this->theta() + theta, this->phi() + phi );
}

Vector& Vector::rotate(double theta) {
	return rotate(theta, 0);
}

Vector& Vector::normalize() {
	return *this = sphericalForm( 1.0, theta(), phi() );
}


//--------------------------------------------------------------------------------
Vector operator *(double k, const Vector &v) {
	return v * k;
}

ostream& operator <<(ostream &out, const Vector &v) {
	return out << '<' << v.x << ", " << v.y << ", " << v.z << '>';
}


} //end of namespace

#endif
