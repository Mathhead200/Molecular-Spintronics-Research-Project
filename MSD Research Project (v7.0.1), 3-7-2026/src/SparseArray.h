#ifndef UDC_HASHMAP
#define UDC_HASHMAP

#include <stdexcept>

namespace udc {

using std::out_of_range;

template <typename T> struct SparseArrayValue {
	bool set;
	T value;

	SparseArrayValue() : set(false), value() { /* empty */ }

	T& create();
	void clear();
};

/*
 * A fixed-sized data structure, which maps (unsigned int -> value type, T).
 * Not all elements need to contain values.
 * Uses std::out_of_range exception.
 */
template <typename T> class SparseArray {
 private:
	unsigned int _capacity;
	SparseArrayValue<T> *values;

 public:
	SparseArray() : _capacity(0), values(NULL) { /* empty */ }
	SparseArray(unsigned int capacity) : _capacity(capacity) { values = new SparseArrayValue<T>[capacity]; }
	~SparseArray() { delete[] values; }

	SparseArray(const SparseArray<T> &);  // do not use: not implemented!
	SparseArray<T>& operator=(const SparseArray<T> &);  // do not use: not implemented!

	unsigned int capacity() const { return _capacity; }
	void resize(unsigned int capacity);  // will clear the array

	// does NO bounds checking
	T& operator[](unsigned int index) { return values[index].create(); }  // creates an element at index if it has not yet been created
	const T& operator[](unsigned int index) const { return values[index].value; }  // Undefined behaviour if index has not been set, or if out of bounds.
	void clear(unsigned int index) { values[index].clear(); }  // has no effect if the element was not yet created

	// DOES bounds checking:
	// throws an out_of_range exception if the given index is out of range or
	// if the element has not yet been created.
	T& at(unsigned int index);
	const T& at(unsigned int index) const;
	void clearAt(unsigned int index);
};

template <typename T> T& SparseArrayValue<T>::create() {
	set = true;
	return value;
}

template <typename T> void SparseArrayValue<T>::clear() {
	set = false;
	value = value();
}

template <typename T> void SparseArray<T>::resize(unsigned int capacity) {
	delete[] values;
	_capacity = capacity;
	values = new SparseArrayValue<T>[capacity];
}

template <typename T> T& SparseArray<T>::at(unsigned int index) {
	if (index < 0 || index >= capacity())
		throw out_of_range("SparseArray::at(unsigned int): illegal index");
	SparseArrayValue<T> &v = values[index];
	if (!v.set)
		throw out_of_range("SparseArray::at(unsigned int): index not yet set");
	return v.value;
}

template <typename T> const T& SparseArray<T>::at(unsigned int index) const {
	if (index < 0 || index >= capacity())
		throw out_of_range("SparseArray::at(unsigned int): illegal index");
	const SparseArrayValue<T> &v = values[index];
	if (!v.set)
		throw out_of_range("SparseArray::at(unsigned int): index not yet set");
	return v.value;
}

template <typename T> void SparseArray<T>::clearAt(unsigned int index) {
	if (index < 0 || index >= capacity())
		throw out_of_range("SparseArray::at(unsigned int): illegal index");
	const SparseArrayValue<T> &v = values[index];
	if (!v.set)
		throw out_of_range("SparseArray::at(unsigned int): index not yet set");
	v.clear();

}

}  // end of namespace udc

#endif
