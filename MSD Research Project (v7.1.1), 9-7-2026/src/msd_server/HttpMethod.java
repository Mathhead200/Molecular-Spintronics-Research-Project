package msd_server;

/**
 * Lists the different standard HTTP Request methods.
 * 
 * @see https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
 * @author Christopher D'Angelo
 * @version December 20, 2023
 */
public enum HttpMethod {

	/**
	 * The GET method requests a representation of the specified resource.
	 * Requests using GET should only retrieve data.
	 */
	GET,
	
	/**
	 * The HEAD method asks for a response identical to a GET request, but
	 * without the response body.
	 */
	HEAD,

	/**
	 * The POST method submits an entity to the specified resource, often
	 * causing a change in state or side effects on the server.
	 */
	POST,
	
	/**
	 * The PUT method replaces all current representations of the target
	 * resource with the request payload.
	 */
	PUT,
	
	/**
	 * The DELETE method deletes the specified resource.
	 */
	DELETE,
	
	/**
	 * The CONNECT method establishes a tunnel to the server identified by
	 * the target resource.
	 */
	CONNECT,
	
	/**
	 * The OPTIONS method describes the communication options for the
	 * target resource.
	 */
	OPTIONS,
	
	/**
	 * The TRACE method performs a message loop-back test along the path to
	 * the target resource.
	 */
	TRACE,
	
	/**
	 * The PATCH method applies partial modifications to a resource.
	 */
	PATCH; 

	/**
	 * Does this method match the given String?
	 * 
	 * @param method
	 * @return <code>true</code> on a case-insensitive match; <code>false</code> otherwise.
	 * @see HttpExchange#getRequestMethod()
	 */
	public boolean is(String method) {
		return method.equalsIgnoreCase(toString());
	}

	/**
	 * Gets the {#link HttpMethod} enum object associated with the given
	 * 	String (case-insensitive).
	 * 
	 * @param method
	 * @return An {@link HttpMethod} object.
	 * @throws IllegalArgumentException If the given String doesn't match
	 * 	one of the valid HttpMethods.
	 */
	public static HttpMethod match(String method) throws UnsupportedException {
		try {
			return HttpMethod.valueOf(method.toUpperCase());
		} catch(IllegalArgumentException ex) {
			throw new UnsupportedException(method, ex);
		}
	}

	public static class UnsupportedException extends RuntimeException {
		private UnsupportedException(String message, Throwable cause) {
			super(message, cause);
		}
	}
}