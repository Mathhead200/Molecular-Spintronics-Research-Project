/**
 * @file Vector.h
 * @author Christopher D'Angelo
 * @brief Contains the udc::Vector class, and related operators.
 * 
 * @version 6.3
 * @date 2024-1-7
 * 
 * @copyright Copyright (c) 2011-2024
 */

#ifndef UDC_VECTOR
#define UDC_VECTOR

#include <cmath>
#include <iostream>
#include "udc.h"


namespace udc {

using std::ostream;

using udc::PI;


/**
 * @brief Stores a basic 3D vector. Vectors are stored internally in rectanglular
 *        form which effects rounding behavior. Each component is stored as a double.
 */
class Vector {
 private:
	static double turn(double angle); // Rotate the angle PI radians; stays in [PI, -PI).

 public:
	/** The x-component of the Vector. Magnatude when projected onto Vector::I. */
	double x;

	/** The y-component of the Vector. Magnatude when projected onto Vector::J. */
	double y;

	/** The z-component of the Vector. Magnatude when projected onto Vector::K. */
	double z;

	Vector(double x, double y, double z);
	Vector(double x, double y);
	Vector();

	static const Vector ZERO;
	static const Vector I;
	static const Vector J;
	static const Vector K;

	static Vector cylindricalForm(double r, double theta, double z);
	static Vector polarForm(double r, double theta);
	static Vector sphericalForm(double rho, double theta, double phi);

	double normSq() const;
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

Vector operator *(double k, const Vector &v);
ostream& operator <<(ostream &out, const Vector &v);


//--------------------------------------------------------------------------------
double Vector::turn(double angle) {
	return angle + (angle <= 0 ? PI : -PI);
}


//--------------------------------------------------------------------------------
/** @brief The zero Vector. All components set to (double) 0.0 */
const Vector Vector::ZERO(0, 0, 0);

/** @brief The first standard basis vector: <x, y, z> = <1, 0, 0> */
const Vector Vector::I(1, 0, 0);

/** @brief The second standard basis vector: <x, y, z> = <0, 1, 0> */
const Vector Vector::J(0, 1, 0);

/** @brief The third standard basis vector: <x, y, z> = <0, 0, 1> */
const Vector Vector::K(0, 0, 1);


//--------------------------------------------------------------------------------
/**
 * @brief Construct a 3D Vector in rectangular form with the given components.
 * 
 * @param x x-component
 * @param y y-component
 * @param z z-component
 */
Vector::Vector(double x, double y, double z) : x(x), y(y), z(z) {
}

/**
 * @brief Construct a 2D Vector in rectangular form with the given components.
 *        z=0
 * 
 * @param x x-component
 * @param y y-component
 */
Vector::Vector(double x, double y) : x(x), y(y), z(0) {
}

/**
 * @brief Construct a (mutable) zero Vector.
 *        For an immutable zero Vector, Vector::ZERO already exists.
 * @see Vector#ZERO
 */
Vector::Vector() : x(0), y(0), z(0) {
}


//-------------------------------------------------------------------------------
/**
 * @brief Construct a 3D Vector in cylindrical form.
 *        The Vector will be converted to rectanular form internally.
 * 
 * @param r Radius of the Vector when projected onto the xy-plane.
 * @param theta Angle (in radians) of the Vector when projected onto the xy-plane. <ul>
 *              <li> 0 (radians) aligns with the positive x-axis. </li>
 *              <li> PI/2 aligns with the positive y-axis. </li>
 *              </ul>
 * @param z Magnatiude of the perpendicular projection onto the xy-plane.
 *          I.e., the height of the vector above (negative if below) the xy-plane.
 * @return Vector
 */
Vector Vector::cylindricalForm(double r, double theta, double z) {
	return Vector( r * cos(theta),  r * sin(theta), z );
}

/**
 * @brief Construct a 2D Vector in polar form. z=0.
 *        The Vector will be converted to rectangular form interanlly.
 * 
 * @param r Magnatude of the Vector.
 * @param theta Angle (in radians) of the Vector. <ul>
 *              <li> 0 (radians) aligns with the positive x-axis. </li>
 *              <li> PI/2 aligns with the positive y-axis. </li>
 *              </ul>
 * @return Vector
 */
Vector Vector::polarForm(double r, double theta) {
	return cylindricalForm(r, theta, 0);
}

/**
 * @brief Constructs a 3D Vector in spherical form.
 *        The Vector will be converted to rectangular form interanlly.
 * 
 * @param rho Magnetude of the Vector.
 * @param theta Angle (in radians) of the Vector when prjected onto the xy-plane. <ul>
 *              <li> 0 (radians) aligns with the positive x-axis. </li>
 *              <li> PI/2 aligns with the positive y-axis. </li>
 *              </ul>
 * @param phi Angle (in radians) of elevation the Vector when projected onto a plane
 *            perpendicular to the xy-plane. <ul>
 *            <li> 0 (radians) aligns with the xy-plane. </li>
 *            <li> PI/2 (radians) aligns with the positive z-axis. </li>
 *            <li> -PI/2 (radians) aligns with the negative z-axis. <li>
 *            </ul>
 *            Note: the details described above for <code>phi</code> do not follow
 *            the standard conventions used in many Math texts.
 * @return Vector
 */
Vector Vector::sphericalForm(double rho, double theta, double phi) {
	return cylindricalForm( rho * cos(phi), theta, rho * sin(phi) );
}


//-------------------------------------------------------------------------------
/**
 * @return the square norm (magnetude) of the Vector. Faster then
 * {@see Vector#norm() const}, so useful when you don't need the actually norm.
 * @see #norm() const
 */
double Vector::normSq() const {
	return x * x + y * y + z * z;
}

/**
 * @return the norm (magnetude) of the Vector.
 * @see #normSq() const
 */
double Vector::norm() const {
	return sqrt( normSq() );
}

/**
 * @return Angle (in radians) of the Vector when projected onto the xy-plane.
 * <ul>
 * 	<li> 0 (radians) aligns with the positive x-axis. </li>
 * 	<li> PI/2 aligns with the positive y-axis. </li>
 * </ul>
 */
double Vector::theta() const {
	return atan2(y, x);
}

/**
 * @return Angle (in radians) of elevation the Vector when projected onto a plane
 * perpendicular to the xy-plane.
 * <ul>
 * 	<li> 0 (radians) aligns with the xy-plane. </li>
 * 	<li> PI/2 (radians) aligns with the positive z-axis. </li>
 * 	<li> -PI/2 (radians) aligns with the negative z-axis. <li>
 * </ul>
 * Note: the details described above for <code>phi</code> do not follow
 * the standard conventions used in many Math texts.
 */
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


/**
 * @brief Compare to see if two Vectors are equal.
 * <p> Note: becasue of floating point rounding errors, two Vectors which should be
 * mathamatically equal, but arived at through different arithmetic algorithms
 * may still be "unequal" under this operator. Therefore, use cases for this operator
 * are very specific. </p>
 * 
 * <p> Consider comparing Vector norms or components with approprate margins of
 * error accounted for. </p>
 * 
 * @param v other Vector
 * @return <code>true</code> if all components are equal;
 * otherwise, <code>false</code>
 */
bool Vector::operator ==(const Vector &v) const {
	return x == v.x && y == v.y && z == v.z;
}

/**
 * @brief Negation of operator ==
 * 
 * @see Vector::operator==(const Vector &) const
 * @param v other Vector
 * @return @return <code>true</code> if any components is unequal;
 * otherwise, <code>false</code> (all components equal)
 */
bool Vector::operator !=(const Vector &v) const {
	return x != v.x || y != v.y || z != v.z;
}


/**
 * @brief Add two Vectors.
 * 
 * @param v other Vector (addend)
 * @return A new, third Vector: the resultant Vector, (i.e., sum)
 */
Vector Vector::operator +(const Vector &v) const {
	return Vector(x + v.x, y + v.y, z + v.z);
}

/**
 * @brief Negate a Vectors.
 * 
 * @return A new Vector which has the same magnetude,
 * but points in the opposite direction.
 */
Vector Vector::operator -() const {
	return Vector(-x, -y, -z);
}

/**
 * @brief Subtract two Vectors.
 * 
 * @param v other Vector (subtrahand)
 * @return A new, third Vector: the difference, <pre>this - v</pre> 
 */
Vector Vector::operator -(const Vector &v) const {
	return Vector(x - v.x, y - v.y, z - v.z);
}

/**
 * @brief Scalar multiplication.
 * Multiply the Vector by the given scalar.
 * 
 * @param k Scalar
 * @return A new Vector parallel to the original, but scaled.
 */
Vector Vector::operator *(double k) const {
	return Vector(k * x, k * y, k * z);
}

/**
 * @brief Dot (inner) product. Multiply two Vectors.
 * 
 * @param v other Vector (multiplier).
 * @return Scalar (double)
 */
double Vector::operator *(const Vector &v) const {
	return dotProduct(v);
}


/**
 * @brief Compute the square distance between two Vectors in standard position.
 * Equivalent to <pre>|v - this|^2</pre>, or <code>(v - *this).normSq()</code>.
 * Faster then computing the actual distance.
 * 
 * @see Vector#distance(const Vector &) const
 * @param v other Vector
 * @return Distance squared (double)
 */
double Vector::distanceSq(const Vector &v) const {
	double dx = v.x - x, dy = v.y - y, dz = v.z - z;
	return dx * dx + dy * dy + dz * dz;
}

/**
 * @brief Computer the distance between two Vectors in standard position.
 * Equivalent to <pre>|v - this|</pre>, or <code>(v - *this).norm()</code>.
 * 
 * @see Vector#distanceSq(const Vector &) const
 * @param v other Vector
 * @return Distance (double) 
 */
double Vector::distance(const Vector &v) const {
	return sqrt( distanceSq(v) );
}

/**
 * @brief Alias for (*this * v)
 * @see Vector#operator *(const Vector &) const
 * @param v other Vector
 * @return Dot product as scalar (double)
 */
double Vector::dotProduct(const Vector &v) const {
	return x * v.x + y * v.y + z * v.z;
}

/**
 * @brief Compute the minor angle between two Vectors.
 * 
 * @param v another Vector
 * @return Angle in radians (double)
 */
double Vector::angleBetween(const Vector &v) const {
	return acos( dotProduct(v) / sqrt(normSq() * v.normSq()) );
}

/**
 * @brief Computer the cross product between two Vectors.
 * 
 * @param v other Vector
 * @return A new, thrid Vector: cross product, perpendicular to the two given
 * vectors, and with a magnatude equal to the product of their magnatudes.
 */
Vector Vector::crossProduct(Vector v) const {
	return Vector( y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x );
}


/**
 * @brief Mutate <code>this</code> Vector by adding another Vector to it.
 * The sum will be stored in <code>this</code>.
 * 
 * @param v other Vector (addend)
 * @return A reference to <code>*this</code>.
 */
Vector& Vector::operator +=(const Vector &v) {
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

/**
 * @brief Mutate <code>this</code> Vector by subtracting another Vector from it.
 * The difference will be stored in <code>this</code>.
 * 
 * @param v other Vector (subtrahand)
 * @return A reference to <code>*this</code>.
 */
Vector& Vector::operator -=(const Vector &v) {
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

/**
 * @brief Mutate <code>this</code> Vector by multiplying it by a scalar.
 * The product will be stored in <code>this</code>.
 * 
 * @param k Scalar (double)
 * @return A reference to <code>*this</code>.
 */
Vector& Vector::operator *=(double k) {
	x *= k;
	y *= k;
	z *= k;
	return *this;
}


/**
 * @brief "flip" <code>this</code> Vector so it faces the other direction.
 * 	Mutates the Vector.
 * 
 * @return A reference to <code>*this</code>.
 */
Vector& Vector::negate() {
	x = -x;
	y = -y;
	z = -z;
	return *this;
}

/**
 * @brief Rotate <code>this</code> Vector by the given angles.
 * 	Mutates the Vector.
 * 
 * @param theta A rotation (radians) in the xy-plane. Positive values rotate in
 * 	the direction starting from the posative x-axis, towards the positive y-axis.
 * @param phi A rotation (radians) perpendicular to the xy-plane. Positive
 * 	values rotate towards the posative z-axis.
 * @return A reference to <code>*this</code>. Useful for chaining.
 */
Vector& Vector::rotate(double theta, double phi) {
	return *this = sphericalForm( norm(), this->theta() + theta, this->phi() + phi );
}

/**
 * @brief Rotate <code>this</code> Vector about the z-axis by the given angle.
 * 	Mutates the Vector.
 * 
 * @see #rotate(double, double)
 * @param theta A rotation (radians) in the xy-plane. Positive values rotate in
 * 	the direction starting from the posative x-axis, towards the positive y-axis.
 * @return A reference to <code>*this</code>. Useful for chaining.
 */
Vector& Vector::rotate(double theta) {
	return rotate(theta, 0);
}

/**
 * @brief Normalize <code>this</code> Vector, after which (unless the Vector is
 * 	the ZERO Vector), its magnatude will be <code>1</code>, and its direction
 * 	will be unchanged.
 * 
 * 	Normalizing the ZERO Vector results in undefined behavior.
 * 
 * @return A reference to <code>*this</code>. Useful for chaining.
 */
Vector& Vector::normalize() {
	double k = 1.0 / norm();
	x *= k;
	y *= k;
	z *= k;
	return *this;
}


//--------------------------------------------------------------------------------
/**
 * @brief Scalar multiplication.
 * 
 * @see Vector::operator *(double) const
 * @param k Scalar (double)
 * @param v Vector
 * @return Product (Vector)
 */
Vector operator *(double k, const Vector &v) {
	return v * k;
}

/**
 * @brief Output this Vector in the form <pre>"<x, y, z>"</pre>
 * 
 * @param out - An output stream (e.g., <code>std::cout</code>)
 * @param v - A Vector object
 * @return The given output stream.
 */
ostream& operator <<(ostream &out, const Vector &v) {
	return out << '<' << v.x << ", " << v.y << ", " << v.z << '>';
}


} //end of namespace

#endif
