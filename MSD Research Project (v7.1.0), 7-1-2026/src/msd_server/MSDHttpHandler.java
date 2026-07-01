package msd_server;

import java.io.IOException;
// import java.util.Date;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;


public interface MSDHttpHandler extends HttpHandler {

	@Override
	default void handle(HttpExchange exchange) {
		HttpResponse res = new HttpResponse(exchange);  // set up HttpResponse object to store response
		try(exchange; res) {
			HttpRequest req = new HttpRequest(exchange);  // parse HttpRequest

			try {
				handle(req, res);  // finish handling the request

			} catch(InterruptedException ex) {
				throw ex;  // skip send()

			} catch(Exception ex) {
				res.setException(ex);
			}
			
			res.send();

		} catch (IOException | HttpMethod.UnsupportedException ex) {
			System.err.println(ex);  // TODO: stub

		} catch(InterruptedException ex) {
			// Do nothing. This is fine.

		}
	}

	/**
	 * 
	 * @param request Contains parsed data about the request
	 * @param response <ol>
	 * 	<li> (Required) Set HTTP Status Code (e.g., 200 OK, or 404 Not Found). </li>
	 *  <li> (Optional) Set any additional response headers.
	 * 		{@link HttpHeader#CONTENT_LENGTH} will be set authomatically based on {@link HttpResponse#body} length. </li>
	 *  <li> (Optional) Set a response body. </li>
	 * @throws Exception
	 */
	public abstract void handle(HttpRequest request, HttpResponse response) throws Exception;
}