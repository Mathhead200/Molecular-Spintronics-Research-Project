package msd_server;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;

public class MSDWorker implements AutoCloseable {
	private Process proc;
	private BufferedReader in;  // streams for python sub-process
	private PrintWriter out;  // streams for python sub-process
	private BufferedReader err;  // streams for python sub-process
	private boolean shutdown = false;
	// TODO: remember to change  "record" to a different object on reset/reinitialize
	// so the old view doesn't get affected or locked
	private ArrayList<String> record = new ArrayList<>(0);
	private StringBuilder errLog = new StringBuilder();
	private Thread errLogReader = new Thread(() -> {
		try {
			while(!Thread.interrupted()) {
				String line = err.readLine();
				if (line == null)
					break;  // Stream closed. Stop listening for new error messages.
				synchronized(errLog) {
					errLog.append(line + "\n");
				}
			}
		} catch(IOException ex) { /* do nothing */ }
	});

	/**
	 * @param state Compact JSON string containing MSD init arguments,
	 * 	config settings, MSD parameters, and Molecule parameters.
	 * @throws IOException
	 */
	public MSDWorker(String args) throws IOException {
		proc = new ProcessBuilder("python", "src/msd_server/MSDWorker.py").start();
		System.out.println("Worker pid=" + proc.pid());  // DEBUG
		in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
		out = new PrintWriter(proc.getOutputStream(), true);
		err = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
		errLogReader.start();  // TODO: not currently using errLog, turn off?

		args = collapse(args);
		out.println(args);
		confirmResponse("READY");
	}

	/**
	 * Blocks while python sub-process is running.
	 * 
	 * @param args Compact JSON string containing: <br>
	 * 	<code>{ simCount, freq, dkT, dB }</code>
	 */
	public void run(String args) throws IOException {
		args = collapse(args);
		synchronized(proc) {
			out.println("RUN");
			out.println(args);
			// TODO: record.ensureCapacity(...)
			String state;
			while(true) {
				state = requireLine();
				if ("DONE".equalsIgnoreCase(state))
					break;
				synchronized(record) {
					record.add(state);
				}

				// is the worker trying to exit?
				if (shutdown || Thread.interrupted()) {
					out.println("CANCEL");
					confirmResponse("DONE");
					shutdown = false;
					break;
				} else {
					out.println("CONTINUE");
				}
			}
		}
	}

	public String getState(String args) throws IOException {
		synchronized(proc) {
			out.println("GET");
			out.println(args);
			return requireLine();
		}
	}

	public String getState() throws IOException {
		// get all state by sending empty JSON object to worker process
		return getState("[]");
	}

	/**
	 * @param parameters Compact JSON string containing MSD parameters
	 */
	public void setParameters(String parameters) throws IOException {
		parameters = collapse(parameters);
		synchronized(proc) {
			out.println("SET");
			out.println(parameters);
			confirmResponse("DONE");
		}
	}

	public String reset(String args) throws IOException {
		synchronized(proc) {
			out.println("RESET");
			out.println(args);
			clearRecord();
			return requireLine();
		}
	}

	public void cancel() {
		shutdown = true;
	}

	public void exit() throws IOException {
		cancel();  // flag signals the run() method in case a simulation is running
		synchronized(proc) {
			out.println("EXIT");
			confirmResponse("GOODBYE");
		}
	}

	public String errLog() {
		synchronized(errLog) {
			return errLog.toString();
		}
	}

	@Override
	public void close() throws IOException {
		try {
			out.close();
			in.close();
			err.close();
		} finally {
			errLogReader.interrupt();
			proc.destroy();
		}
	}

	/**
	 * @return A synchronized List backed by the underlying MSDWorker.
	 *         The list size is potentially volutile, but should never decrease.
	 */
	public List<String> getRecord() {
		var record = this.record;  // copy reference in-case of reset/reinitialize
		return new AbstractList<>() {
			@Override
			public int size() {
				synchronized(record) {
					return record.size();
				}
			}

			@Override
			public String get(int index) {
				synchronized(record) {
					return record.get(index);
				}
			}
		};
	}

	private void clearRecord() {
		synchronized(record) {
			record = new ArrayList<>(0);
		}
	}

	/**
	 * Removes all unquoted whitespace.
	 * @param json
	 * @return Minified JSON
	 * @see https://www.baeldung.com/java-json-minify-remove-whitespaces
	 */
	private String collapse(String json) {
		StringBuilder result = new StringBuilder(json.length());
		boolean inQuotes = false;
		boolean escapeMode = false;
		for (char character : json.toCharArray()) {
			if (escapeMode) {
				result.append(character);
				escapeMode = false;
			} else if (character == '"') {
				inQuotes = !inQuotes;
				result.append(character);
			} else if (character == '\\') {
				escapeMode = true;
				result.append(character);
			} else if (!inQuotes && Character.isWhitespace(character)) {
				continue;
			} else {
				result.append(character);
			}
		}
		return result.toString();
	}

	/**
	 * Read the next response from the sub-prcocess, and make sure it matches the
	 * given String.
	 * 
	 * @param expectedResponse Expected next response from server.
	 * @throws WorkerSyncException The sub-process responded with a different
	 *                             String, or null (end-of-stream).
	 * @throws IOException         A different error occurs while reading.
	 */
	private void confirmResponse(String expectedResponse) throws IOException {
		String line = in.readLine();
		if (!expectedResponse.equalsIgnoreCase(line)) {
			throw new WorkerSyncException(String.format(
				"expected \"%s\", but received \"%s\"",
				expectedResponse, line ));
		}
	}

	/**
	 * Reads a line from sub-process's standard input.
	 * 
	 * @return A non-null response from the sub-process.
	 * @throws WorkerSyncException If null is read.
	 * @throws IOException         If a different error occurses while reading.
	 */
	private String requireLine() throws IOException {
		String line = in.readLine();
		if  (line == null)
			throw new WorkerSyncException("Unexpected end of stream when reading from worker.");
		return line;
	}
}
