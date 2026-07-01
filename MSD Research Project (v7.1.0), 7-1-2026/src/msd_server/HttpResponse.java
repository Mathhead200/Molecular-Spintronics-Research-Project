package msd_server;

import static msd_server.MSDServer.encode;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

/**
 * @author Christopher D'Angelo
 * @version December 29, 2023
 */
public class HttpResponse implements AutoCloseable {
	private HttpExchange exchange;
	private WritableByteChannel bodyOut;
	private boolean writeUsed = false;

	public HttpStatus status;
	public Headers headers;
	public ByteBuffer body = null;

	public HttpResponse(HttpExchange exchange) {
		this.exchange = exchange;
		this.bodyOut = Channels.newChannel(exchange.getResponseBody());
		this.headers = exchange.getResponseHeaders();
		// TODO: Set some default headers?
		// HttpHeader.DATE.to(new Date().toGMTString(), response.headers);
	}

	public int setBody(CharBuffer str) {
		body = encode(str);
		return body.position();
	}

	public int setBody(CharSequence str) {
		return setBody(CharBuffer.wrap(str));
	}

	public void setException(Throwable ex) {
		status = HttpStatus.INTERNAL_SERVER_ERROR;
		StringWriter trace = new StringWriter();
		ex.printStackTrace(new PrintWriter(trace));
		setBody(String.format("""
			<html>
				<body>
					<p>%s</p> <br>
					<pre>%s</pre>
				</body>
			</html>
		""", status, trace));
		HttpHeader.CONTENT_TYPE.to("text/html; utf-8", headers);
	}

	/**
	 * Writes {@link #headers} and optionally a {@link #body} if it has been set.
	 * @throws IOException
	 * @see #setBody(CharBuffer)
	 */
	public void send() throws IOException {
		if (writeUsed)
			return;  // don't allow sends following any write...() method
		if (body != null) {
			exchange.sendResponseHeaders(status.code, body.position());
			bodyOut.write(body);
		} else {
			exchange.sendResponseHeaders(status.code, -1);
		}
	}

	/**
	 * Writes the headers stored in {@link #headers}.
	 * Uses chunked encoding since length is unknown at this time.
	 * 
	 * @see #writeBody(CharBuffer)
	 * @see HttpExchange#sendResponseHeaders(int, long)
	 */
	public void writeHeaders() throws IOException {
		writeUsed = true;
		exchange.sendResponseHeaders(status.code, 0);
	}

	/**
	 * Writes raw bytes to the body of the HTTP Response. Used in place of
	 * {@link #send()}. Must call {@link #writeHeaders()} first. After a call to
	 * this method, any subsiquent call to send will do nothing.
	 * 
	 * @param buf Binary data to send.
	 * @throws IOException
	 * @see {@link #writeBody(CharBuffer)}
	 */
	public void writeBody(ByteBuffer buf) throws IOException {
		writeUsed = true;
		bodyOut.write(buf);
	}

	/**
	 * Write to the body of the HTTP Response. Used in place of {@link #send()}.
	 * Must call {@link #writeHeaders()} first. After a call to this method, any
	 * subsiquent call to send will do nothing.
	 * 
	 * @param str String to send.
	 * @return Number of bytes written after encoding.
	 * @throws IOException
	 */
	public int writeBody(CharBuffer str) throws IOException {
		ByteBuffer buf = encode(str);
		int length = buf.position();
		writeBody(buf);
		return length;
	}

	/**
	 * Convenience method.
	 * @see #writeBody(CharBuffer)
	 */
	public int writeBody(CharSequence str) throws IOException {
		return writeBody(CharBuffer.wrap(str));
	}

	@Override
	public void close() throws IOException {
		bodyOut.close();
	}
}
