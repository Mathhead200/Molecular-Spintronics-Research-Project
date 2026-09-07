package msd_server;

import static msd_server.MSDServer.decode;
import static msd_server.MSDServer.decodeURL;

import java.io.IOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

/**
 * @author Christopher D'Angelo
 * @version December 29, 2023
 */
public class HttpRequest {
	private static final float QUERY_HASH_LOAD_FACTOR = 0.5f;

	public HttpMethod method;
	public String path;
	public Map<String, String> query;
	public String fragment;
	public Headers headers;
	public byte[] body;

	/**
	 * Parses and adds helper methods to the following {@link HttpExchange} object.
	 * @param exchange
	 * @throws HttpMethod.UnsupportedException
	 * @throws IOException
	 */
	public HttpRequest(HttpExchange exchange) throws IOException, HttpMethod.UnsupportedException {
		// parse HTTP method
		this.method = HttpMethod.match(exchange.getRequestMethod());

		URI uri = exchange.getRequestURI();

		// parse HTTP Requested URI path
		this.path = uri.getPath();

		// parse URI ?query=string
		String q = uri.getRawQuery();
		if (q != null) {
			String[] pairs = q.split("&");
			this.query = new HashMap<String, String>(
				(int) Math.ceil(pairs.length / QUERY_HASH_LOAD_FACTOR),
				QUERY_HASH_LOAD_FACTOR );
			for (String pair : pairs) {
				String[] kv = pair.split("=", 2);  // { key, value }
				if (kv.length == 2)
					this.query.put(decodeURL(kv[0]), decodeURL(kv[1]));
				else  // kv.length == 1
					this.query.put(decodeURL(kv[0]), null);
			}
		}

		// parse URI #fragment (i.e., HTML #anchor)
		this.fragment = uri.getFragment();

		// parse HTTP Request headers
		this.headers = exchange.getRequestHeaders();
		
		// parse HTTP Request body
		if (HttpHeader.CONTENT_LENGTH.in(this.headers)) {
			this.body = new byte[HttpHeader.CONTENT_LENGTH.from(this.headers, Integer::parseInt)];
			exchange.getRequestBody().read(this.body, 0, this.body.length);
		}
	}

	public String getBody() {
		return decode(ByteBuffer.wrap(body)).toString();
	}

	public String requireQueryParameter(String name, HttpResponse res) throws RequiredException {
		if (query == null || !query.containsKey(name)) {
			res.status = HttpStatus.BAD_REQUEST;
			res.setBody("Missing a required query parameter: " + name);
			HttpHeader.CONTENT_TYPE.to("text/plain; utf-8", res.headers);
			throw new RequiredException();
		}
		return query.get(name);
	}
}
