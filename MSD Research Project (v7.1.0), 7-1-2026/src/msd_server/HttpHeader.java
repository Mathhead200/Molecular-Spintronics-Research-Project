package msd_server;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

import com.sun.net.httpserver.Headers;

/**
 * List of standard HTTP Headers.
 * 
 * @see https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers
 * @author Christopher D'Angelo
 * @version December 29, 2023
 */
public enum HttpHeader {
 // [Authentication]
	/**
	 * Defines the authentication method that should be used to access a resource.
	 */
	WWW_AUTHENTICATE("WWW-Authenticate"),

	/**
	 * Contains the credentials to authenticate a user-agent with a server.
	 */
	AUTHORIZATION("Authorization"),

	/**
	 * Defines the authentication method that should be used to access a
	 * resource behind a proxy server.
	 */
	PROXY_AUTHENTICATE("Proxy-Authenticate"),

	/**
	 * Contains the credentials to authenticate a user agent with a proxy server.
	 */
	PROXY_AUTHORIZATION("Proxy-Authorization"),

 // [Caching]
	/**
	 * The time, in seconds, that the object has been in a proxy cache.
	 */
	AGE("Age"),

	/**
	 * Directives for caching mechanisms in both requests and responses.
	 */
	CACHE_CONTROL("Cache-Control"),

	/**
	 * Clears browsing data (e.g. cookies, storage, cache) associated
	 * with the requesting website.
	 */
	CLEAR_SITE_DATA("Clear-Site-Data"),

	/**
	 * The date/time after which the response is considered stale.
	 */
	EXPIRES("Expires"),

	// [Conditionals]
	/**
	 * The last modification date of the resource, used to compare several
	 * versions of the same resource. It is less accurate than ETag, but
	 * easier to calculate in some environments. Conditional requests using
	 * If-Modified-Since and If-Unmodified-Since use this value to change
	 * the behavior of the request.
	 */
	LAST_MODIFIED("Last-Modified"),

	/**
	 * A unique string identifying the version of the resource. Conditional
	 * requests using If-Match and If-None-Match use this value to change
	 * the behavior of the request.
	 */
	ETag("ETag"),

	/**
	 * Makes the request conditional, and applies the method only if the
	 * stored resource matches one of the given ETags.
	 */
	IF_MATCH("If-Match"),

	/**
	 * Makes the request conditional, and applies the method only if the
	 * stored resource doesn't match any of the given ETags. This is used
	 * to update caches (for safe requests), or to prevent uploading a new
	 * resource when one already exists.
	 */
	IF_NONE_MATCH("If-None-Match"),

	/**
	 * Makes the request conditional, and expects the resource to be
	 * transmitted only if it has been modified after the given date. This
	 * is used to transmit data only when the cache is out of date.
	 */
	IF_MODIFIED_SINCE("If-Modified-Since"),

	/**
	 * Makes the request conditional, and expects the resource to be
	 * transmitted only if it has not been modified after the given date.
	 * This ensures the coherence of a new fragment of a specific range
	 * with previous ones, or to implement an optimistic concurrency
	 * control system when modifying existing documents.
	 */
	IF_UNMODIFIED_SINCE("If-Unmodified-Since"),

	/**
	 * Determines how to match request headers to decide whether a cached
	 * response can be used rather than requesting a fresh one from the
	 * origin server.
	 */
	VARY("Vary"),

 // [Connection management]
	/**
	 * Controls whether the network connection stays open after the current
	 * transaction finishes.
	 */
	CONNECTION("Connection"),

	/**
	 * Controls how long a persistent connection should stay open.
	 */
	KEEP_ALIVE("Keep-Alive"),

	// [Content negotiation]
	/**
	 * Informs the server about the types of data that can be sent back.
	 */
	ACCEPT("Accept"),

	/**
	 * The encoding algorithm, usually a compression algorithm, that can be
	 * used on the resource sent back.
	 */
	ACCEPT_ENCODING("Accept-Encoding"),

	/**
	 * Informs the server about the human language the server is expected
	 * to send back. This is a hint and is not necessarily under the full
	 * control of the user: the server should always pay attention not to
	 * override an explicit user choice (like selecting a language from a dropdown).
	 */
	ACCEPT_LANGUAGE("Accept-Language"),

 // [Controls]
	/**
	 * Indicates expectations that need to be fulfilled by the server to
	 * properly handle the request.
	 */
	EXPECT("Expect"),

	/**
	 * When using TRACE, indicates the maximum number of hops the request
	 * can do before being reflected to the sender.
	 */
	MAX_FORWARDS("Max-Forwards"),

 // [Cookies]
	/**
	 * Contains stored HTTP cookies previously sent by the server with the
	 * Set-Cookie header.
	 */
	COOKIE("Cookie"),

	/**
	 * Send cookies from the server to the user-agent.
	 */
	SET_COOKIE("Set-Cookie"),

 // [CORS] (TODO: docs)
	ACCESS_CONTROL_ALLOW_CREDENTIALS("Access-Control-Allow-Credentials"),
	ACCESS_CONTROL_ALLOW_HEADERS("Access-Control-Allow-Headers"),
	ACCESS_CONTROL_ALLOW_METHODS("Access-Control-Allow-Methods"),
	ACCESS_CONTROL_ALLOW_ORIGIN("Access-Control-Allow-Origin"),
	ACCESS_CONTROL_EXPOSE_HEADERS("Access-Control-Expose-Headers"),
	ACCESS_CONTROL_MAX_AGE("Access-Control-Max-Age"),
	ACCESS_CONTROL_REQUEST_HEADERS("Access-Control-Request-Headers"),
	ACCESS_CONTROL_REQUEST_METHOD("Access-Control-Request-Method"),
	Origin("Origin"),
	TIMING_ALLOW_ORIGIN("Timing-Allow-Origin"),

 // TODO: [Downloads]

 // [Message body information]
	/**
	 * The size of the resource, in decimal number of bytes.
	 */
	CONTENT_LENGTH("Content-Length"),

	/**
	 * Indicates the media type of the resource.
	 */
	CONTENT_TYPE("Content-Type"),

	/**
	 * Used to specify the compression algorithm.
	 */
	CONTENT_ENCODING("Content-Encoding"),

	/**
	 * Describes the human language(s) intended for the audience, so that it
	 * allows a user to differentiate according to the users' own preferred language.
	 */
	CONTENT_LANGUAGE("Content-Language"),

	/**
	 * Indicates an alternate location for the returned data.
	 */
	CONTENT_LOCATION("Content-Location"),

 // TODO: [Proxies]

 // TODO: [Redirects]
	/**
	 * <p>
	 * The Location response header indicates the URL to redirect a page to. It only
	 * provides a meaning when served with a <code>3xx</code> (redirection) or
	 * <code>201</code> (created)
	 * status response.
	 * </p>
	 * 
	 * <p>
	 * In cases of redirection, the HTTP method used to make the new request to
	 * fetch the page pointed to by Location depends on the original method and the
	 * kind of redirection:/p>
	 * <ul>
	 * <li><code>303</code> ({@link HttpStatus#SEE_OTHER}) responses always lead to
	 * the use of a GET method.</li>
	 * <li><code>307</code> ({@link HttpStatus#TEMPORARY_REDIRECT}) and
	 * <code>308</code> ({@link HttpStatus#PERMANENT_REDIRECT}) don't change the
	 * method used in the original request.</li>
	 * <li><code>301</code> ({@link HttpStatus#MOVED_PERMANENTLY}) and
	 * <code>302</code> ({@link HttpStatus#FOUND}) don't change the method most of
	 * the time, though older user-agents may (so you basically don't know).</li>
	 * <li>All responses with one of these status codes send a Location header.</li>
	 * </ul>
	 * <p>
	 * In cases of resource creation, it indicates the URL to the newly created
	 * resource.
	 * </p>
	 * 
	 * <p>
	 * Location and {@link #CONTENT_LOCATION} are different. Location indicates the
	 * target of
	 * a redirection or the URL of a newly created resource.
	 * {@link #CONTENT_LOCATION}
	 * indicates the direct URL to use to access the resource when content
	 * negotiation happened, without the need of further content negotiation.
	 * Location is a header associated with the response, while
	 * {@link #CONTENT_LOCATION} is
	 * associated with the entity returned.
	 * </p>
	 */
	LOCATION("Location"),

	/**
	 * Directs the browser to reload the page or redirect to another. Takes the same
	 * value as the meta element with <a href=
	 * "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/meta#http-equiv">http-equiv="refresh"</a>.
	 */
	REFRESH("Refresh"),


 // [Request context]
	/**
	 * Contains an Internet email address for a human user who controls the
	 * requesting user agent.
	 */
	FROM("From"),

	/**
	 * Specifies the domain name of the server (for virtual hosting), and
	 * (optionally) the TCP port number on which the server is listening.
	 */
	HOST("Host"),

	/**
	 * The address of the previous web page from which a link to the currently
	 * requested page was followed.
	 */
	REFERER("Referer"),

	/**
	 * Governs which referrer information sent in the Referer header should be
	 * included with requests made.
	 */
	REFERRER_POLICY("Referrer-Policy"),

	/**
	 * Contains a characteristic string that allows the network protocol peers
	 * to identify the application type, operating system, software vendor or
	 * software version of the requesting software user agent.
	 */
	USER_AGENT("User-Agent"),

 // [Response context]
	/**
	 * Lists the set of HTTP request methods supported by a resource.
	 */
	ALLOW("Allow"),

	/**
	 * Contains information about the software used by the origin server to handle
	 * the request.
	 */
	SERVER("Server"),

 // TODO: [Range requests]

 // TODO: [Security]

 // TODO: [Server-sent events]

 // TODO: [Transfer coding]

 // [Other]
	/**
	 * Used to list alternate ways to reach this service.
	 */
	ALT_SVC("Alt-Svc"),

	/**
	 * Used to identify the alternative service in use.
	 */
	ALT_USED("Alt-Used"),

	/**
	 * Contains the date and time at which the message was originated.
	 */
	DATE("Date"),

	/**
	 * This entity-header field provides a means for serializing one or more
	 * links in HTTP headers. It is semantically equivalent to the HTML <link> element.
	 */
	LINK("Link"),

	RETRY_AFTER("Retry-After"),

	SERVER_TIMING("Server-Timing"),

	SERVICE_WORKER_ALLOWED("Service-Worker-Allowed"),

	SOURCEMAP("SourceMap"),

	UPGRADE("Upgrade"),

 // TODO: [Experimental headers]

 // TODO: [Non-standard headers]

 // TODO: [Deprecated headers]
	
	;


	private final String str;

	HttpHeader(String header) {
		this.str = header;
	}

	/**
	 * Converts a value to a one element list of same type which contains that object.
	 * @param <T>
	 * @param val An object
	 * @return A List containing only the given object.
	 */
	private static <T> List<T> list(T val) {
		List<T> list = new LinkedList<>();
		list.add(val);
		return list;
	}

	@Override
	public String toString() {
		return this.str;
	}

	public List<String> allFrom(Headers map) {
		return map.get(toString());
	}

	public <T> List<T> allFrom(Headers map, Function<String, T> f) {
		List<String> list = allFrom(map);
		if (list == null)
			return null;
		return list.stream().map(f).toList();
	}

	public String from(Headers map) {
		List<String> list = allFrom(map);
		if (list == null)
			return null;
		return list.iterator().next();
	}

	public <T> T from(Headers map, Function<String, T> transform) {
		return transform.apply(from(map));
	}

	public boolean in(Headers map) {
		return map.containsKey(toString());
	}

	public void to(String value, Headers map) {
		if(in(map))
			map.get(toString()).add(value);
		else
			map.put(toString(), list(value));
	}

	public void allTo(Collection<String> values, Headers map) {
		if (in(map))
			map.get(toString()).addAll(values);
		else
			map.put(toString(), new ArrayList<>(values));
	}
}
