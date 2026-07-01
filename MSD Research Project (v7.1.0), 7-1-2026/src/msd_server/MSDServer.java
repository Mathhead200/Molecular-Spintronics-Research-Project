package msd_server;

import static msd_server.HttpStatus.OK;
import static msd_server.HttpStatus.CREATED;
import static msd_server.HttpStatus.ACCEPTED;
import static msd_server.HttpStatus.NO_CONTENT;
import static msd_server.HttpStatus.BAD_REQUEST;
import static msd_server.HttpStatus.NOT_FOUND;
import static msd_server.HttpStatus.METHOD_NOT_ALLOWED;

import static msd_server.HttpHeader.ACCESS_CONTROL_ALLOW_ORIGIN;
import static msd_server.HttpHeader.ACCESS_CONTROL_ALLOW_METHODS;
import static msd_server.HttpHeader.ACCESS_CONTROL_ALLOW_HEADERS;
import static msd_server.HttpHeader.ACCESS_CONTROL_MAX_AGE;
import static msd_server.HttpHeader.ACCESS_CONTROL_REQUEST_METHOD;
import static msd_server.HttpHeader.CONTENT_TYPE;
import static msd_server.HttpHeader.LOCATION;

import static msd_server.HttpMethod.GET;
import static msd_server.HttpMethod.POST;
import static msd_server.HttpMethod.PATCH;
import static msd_server.HttpMethod.DELETE;
import static msd_server.HttpMethod.OPTIONS;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.EnumSet;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import com.sun.net.httpserver.HttpServer;

public class MSDServer {
	public static final String PROTOCOL = "http";  // TODO: support https ??
	public static final Charset CHARSET = StandardCharsets.UTF_8;
	
	public static final int KB = 1024;
	public static final int MB = 1024 * KB;
	public static final int GB = 1024 * MB;

	public static InetSocketAddress address = null;
	public static final WorkerPool workers = new WorkerPool();

	/**
	 * Used for running asynchronous jobs on this server. Namely:
	 * {@link MSDWorker#run(String)}
	 */
	public static ExecutorService threadPool = Executors.newCachedThreadPool();

	public static String decodeURL(String str) {
		return URLDecoder.decode(str, CHARSET);
	}

	public static String encodeURL(String str) {
		return URLEncoder.encode(str, CHARSET);
	}

	public static CharBuffer decode(ByteBuffer buf) {
		return CHARSET.decode(buf);
	}

	public static ByteBuffer encode(CharBuffer str) {
		return CHARSET.encode(str);
	}

	public static ByteBuffer encode(CharSequence str) {
		return encode(CharBuffer.wrap(str));
	}

	/**
	 * @param req HTTP Request
	 * @param res HTTP Response
	 * @param allowedMethods Set of allowed HTTP Methods
	 * @see https://www.baeldung.com/cs/cors-preflight-requests#:~:text=A%20preflight%20request%20is%20an,HTTP%20method%20the%20request%20implements
	*/
	private static void allowCors(HttpResponse res, EnumSet<HttpMethod> allowedMethods) {
		ACCESS_CONTROL_ALLOW_ORIGIN.to("*", res.headers);
		ACCESS_CONTROL_ALLOW_METHODS.to(allowedMethods.stream().map(HttpMethod::toString).collect(Collectors.joining(",")), res.headers);
		ACCESS_CONTROL_ALLOW_HEADERS.to("*", res.headers);
		ACCESS_CONTROL_MAX_AGE.to("86400", res.headers);
	}

	private static void handleOptionsRequest(HttpRequest req, HttpResponse res, EnumSet<HttpMethod> allowedMethods) {
		try {
			if (!allowedMethods.containsAll(ACCESS_CONTROL_REQUEST_METHOD.allFrom(req.headers, HttpMethod::match))) {
				res.status = METHOD_NOT_ALLOWED;
				return;
			}
		} catch(HttpMethod.UnsupportedException | NullPointerException ex) {
			res.status = BAD_REQUEST;  // OPTIONS request must have ACCESS_CONTROL_REQUEST_METHOD
			return;
		}

		res.status = NO_CONTENT;
	}

	/**
	 * Path: /msd
	 * Methods: POST, PATCH, GET, DELETE, OPTIONS
	 */
	private static final MSDHttpHandler msdHandler = (req, res) -> {
		final EnumSet<HttpMethod> allowedMethods = EnumSet.of(POST, PATCH, GET, DELETE, OPTIONS);
		allowCors(res, allowedMethods);

		switch(req.method) {
		case OPTIONS: {
			handleOptionsRequest(req, res, allowedMethods);
		} break;

		case POST: {
			/*
			* Summary: Create a new MSDWorker.
			* 
			* Takes JSON containing MSD.init (required) and
			* 	MSD.parameters (optional) values
			* Returns MSD ID for the newly created MSDWorker.
			*/
			String id = workers.createWorker(req.getBody()).toString();
			res.setBody(String.format("{\"id\":\"%s\"}", id));
			CONTENT_TYPE.to("application/json", res.headers);
			LOCATION.to( String.format("%s://%s:%s/msd?id=%s",
				PROTOCOL, address.getHostString(), address.getPort(), id
				), res.headers );
			res.status = CREATED;
		} break;

		case GET: {
			/*
			* Summary: Check the state of the given MSDWorker.
			* 
			* Takes query-param id=[MSD ID]
			* Returns state of the MSD Worker.
			*/
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;

			res.setBody(msd.getState());
			CONTENT_TYPE.to("application/json", res.headers);
			res.status = OK;
		} break;
		
		case PATCH: {
			/*
			* Summary: Change MSD.parameters for the given MSDWorker.
			* 
			* Takes query-param id=[MSD ID].
			* Takes JSON containing MSD.parameter values to update.
			* Returns empty JSON.
			*/
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;

			msd.setParameters(req.getBody());
			res.status = NO_CONTENT;
		} break;
		
		case DELETE: {
			/*
			* Summary: Destroy the given MSDWorker.
			* 
			* Takes query-param id=[MSD ID].
			* Returns empty HTTP body.
			*/
			MSDWorker msd = workers.destroyWorker(req, res);
			if (msd == null)  break;  // don't set HTTP status to 200 OK

			res.status = NO_CONTENT;
		} break;
		
		default:
			res.status = METHOD_NOT_ALLOWED;
		}
	};

	/**
	 * Path: /run
	 * Methods: POST, DELETE, OPTIONS
	 */
	private static final MSDHttpHandler runHandler = (req, res) -> {
		final EnumSet<HttpMethod> allowedMethods = EnumSet.of(POST, DELETE, OPTIONS);
		allowCors(res, allowedMethods);

		switch(req.method) {
		case OPTIONS: {
			handleOptionsRequest(req, res, allowedMethods);
		} break;

		case POST: {
			/*
			 * Summary: Runs the metropolis algorith.
			 * 
			 * HTTP Request body:
			 * {
			 * 	"simCount": int,  // (Required) The number of iterations to simulate
			 *  "freq": int,      // (Optional) how frequently to record the MSD state, measured in iterations
			 *  "dkT": double,    // (Optional) Heat or cool the system linearly at the given rate (per 1 iteration)
			 *  "dB": Vector      // (Optional) Change the magnetic field linearly at the given rete (per 1 iteration)
			 * }
			 */
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;
			
			threadPool.submit(() -> {
				try {
					System.out.printf("Running simulation %s...%n", req.query.get("id"));  // TODO: DEBUG
					msd.run(req.getBody());
					System.out.printf("Finished simulation %s...%n", req.query.get("id"));  // TODO: DEBUG
				} catch(IOException ex) {
					ex.printStackTrace();  // TODO: stub
				}
			});
			LOCATION.to( String.format("%s://%s:%s/msd/record?id=%s",
				PROTOCOL, address.getHostString(), address.getPort(), req.query.get("id")
				), res.headers );
			res.status = ACCEPTED;
		} break;

		case DELETE: {
			/*
			 * Summary: Cancel the simualtion if it is currently running.
			 */
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;

			msd.cancel();
			res.status = OK;
		} break;

		default:
			res.status = METHOD_NOT_ALLOWED;
		}
	};

	/**
	 * Path: /results
	 * Method: GET, OPTIONS
	 */
	private static final MSDHttpHandler resultsHandler = (req, res) -> {
		final EnumSet<HttpMethod> allowedMethods = EnumSet.of(GET, OPTIONS);
		final int MAX_RESPONSE_SIZE = 1 * GB;
		allowCors(res, allowedMethods);

		switch(req.method) {
		case OPTIONS: {
			handleOptionsRequest(req, res, allowedMethods);
		} break;

		case GET: {
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;

			List<String> record = msd.getRecord();
			int length = record.size();
			if (req.query != null && req.query.containsKey("all")) {
				// Query parameters: all
				// Gets all the available state data. Might be too large!
				CONTENT_TYPE.to("application/json", res.headers);
				res.status = OK;
				res.writeHeaders();

				// write JSON array of state up until MAX_RESONSE_SIZE
				Predicate<Integer> shouldWrite = new Predicate<>() {
					int dataSent = 0;  // data sent so far measured in bytes
					@Override
					public boolean test(Integer n) {
						return (dataSent += n) < MAX_RESPONSE_SIZE - 2;  // -2 for surrounding brackets [ ]
					}
				};
				res.writeBody("[");
				for (int i = 0; i < length; i++) {
					ByteBuffer buf = encode(record.get(i));
					if (!shouldWrite.test(buf.position() + 1))  // +1 for comma ,
						break;
					res.writeBody(buf);
					res.writeBody(",");
				}
				res.writeBody("]");

			} else if (req.query != null && req.query.containsKey("start")) {
				// Query parameters: start=[int] & end=[int]
				// Gets a range of available state data.
				try {
					int start = Integer.valueOf(req.query.get("start"));
					int end = Integer.valueOf(req.requireQueryParameter("end", res));
					
					// allow negative indices
					if (start < 0)  start += length;
					if (end < 0)  end += length;

					CONTENT_TYPE.to("application/json", res.headers);
					res.status = OK;
					res.writeHeaders();

					res.writeBody("[");
					if (start < end)
						res.writeBody(record.get(start));
					for (int i = start + 1; i < end; i++) {
						res.writeBody(",");
						res.writeBody(record.get(i));
					}
					res.writeBody("]");

				} catch(IndexOutOfBoundsException | NumberFormatException ex) {
					res.status = NOT_FOUND;
				
				} catch(RequiredException ex) {
					res.status = BAD_REQUEST;
				}

			} else if (req.query != null && req.query.containsKey("index")) {
				// Query parameters: index=[int]
				// Gets one instance of state data 
				try {
					int index = Integer.valueOf(req.query.get("index"));
					if (index < 0)  index += length;  // allow negative indices
					res.setBody(record.get(index));
					CONTENT_TYPE.to("application/json", res.headers);
					res.status = OK;
				
				} catch(IndexOutOfBoundsException | NumberFormatException ex) {
					res.status = NOT_FOUND;
				}

			} else {
				// Query parameter: (None)
				// Gets the number of state data instances available, indexed 0 through length-1 (inclusive)
				res.setBody(String.format("{\"length\":%s}", length));
				CONTENT_TYPE.to("application/json", res.headers);
				res.status = OK;
			}
		} break;

		default:
			res.status = METHOD_NOT_ALLOWED;
		}
	};

	/**
	 * Path: /reset
	 * Method: POST, OPTIONS
	 */
	private static final MSDHttpHandler resetHandler = (req, res) -> {
		final EnumSet<HttpMethod> allowedMethods = EnumSet.of(POST, OPTIONS);
		allowCors(res, allowedMethods);

		switch(req.method) {
		case OPTIONS: {
			handleOptionsRequest(req, res, allowedMethods);
		} break;

		case POST: {
			MSDWorker msd = workers.lookup(req, res);
			if (msd == null)  break;

			res.setBody(msd.reset(req.getBody()));
			CONTENT_TYPE.to("application/json", res.headers);
			res.status = OK;
		} break;

		default:
			res.status = METHOD_NOT_ALLOWED;
		}
	};

	public static void main(String[] args) throws IOException {
		// Parse cmd args
		if (args.length == 0)
			address = new InetSocketAddress("localhost", 8080);
		else if (args.length == 1)
			address = new InetSocketAddress(Integer.parseInt(args[0]));
		else
			address = new InetSocketAddress(args[0], Integer.parseInt(args[1]));

		// Set up HTTP server
		HttpServer server = HttpServer.create(address, 0);
		server.setExecutor(Executors.newCachedThreadPool());

		server.createContext("/msd", msdHandler);
		server.createContext("/run", runHandler);
		server.createContext("/results", resultsHandler);
		server.createContext("/reset", resetHandler);
		
		// Start server
		server.start();
		System.out.println("MSD Server started.");  // DEBUG

		// Allow for server commands through System.in
		try(Scanner stdin = new Scanner(System.in)) {
			// ignore all commands other then "shutdown"
			while (!stdin.nextLine().equalsIgnoreCase("shutdown")) { /* do nothing */ }
			
			// shutdown
			System.out.println("Shutting down simulations...");
			threadPool.shutdown();
			System.out.println("Deleting data...");
			workers.close();
			System.out.println("Shutting down server...");
			server.stop(10);
			System.out.println("Server successfully closed.");
			// TODO: some thread is still running. Maybe MSDWorker.errLogReader ??
		}
	}
}