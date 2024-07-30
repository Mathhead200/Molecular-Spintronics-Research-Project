package msd_server;

import java.io.IOException;
import java.util.ArrayList;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class WorkerPool implements AutoCloseable {
	private ConcurrentHashMap<UUID, MSDWorker> workers = new ConcurrentHashMap<>();
	private final Object createWorkerLock = new Object();

	public UUID createWorker(String json) throws IOException {
		// create worker
		MSDWorker worker = new MSDWorker(json);

		// generate random worker ID, and add to map
		UUID id;
		synchronized(createWorkerLock) {
			do {
				id = UUID.randomUUID();
			} while(workers.containsKey(id));

			workers.put(id, worker);
		}

		// return the new worker's ID
		return id;
	}

	/**
	 * @param id
	 * @return <code>null</code> if no worker with that ID exists; otherwise, the
	 *         {@link MSDWorker} object created with the given id.
	 */
	public MSDWorker lookup(UUID id) {
		return id != null ? workers.get(id) : null;
	}

	/**
	 * Conveniance method which extracts the worker id from the given {@link HttpRequest}.
	 * If the request doesn't contain an id field, or if the id is invalid, the given {@link HttpResponse}
	 * object is confirgured to send an appropriate error message. 
	 * @param req Request containing worker id
	 * @param res Modified only if request is invalid
	 * @return The {@link MSDWorker} associated with the given id, or <code>null</code> if an error occured.
	 * @see #lookup(String)
	 */
	public MSDWorker lookup(HttpRequest req, HttpResponse res) {
		try {
			return requireAction( this::lookup, requireId(req, res), res );
		} catch(RequiredException ex) {
			return null;
		} catch(IOException ex) {
			throw new Error(ex);  // this can't happen becasue lookup() can't throw IOException
		}
	}

	public MSDWorker destroyWorker(UUID id) throws IOException {
		MSDWorker msd = workers.remove(id);
		if (msd == null)
			return null;
		try(msd) {  // close() MSDWorker after exit()
			msd.exit();
		}
		return msd;
	}

	public MSDWorker destroyWorker(HttpRequest req, HttpResponse res) throws IOException {
		try {
			return requireAction( this::destroyWorker, requireId(req, res), res );
		} catch(RequiredException ex) {
			return null;
		}
	}

	// TODO: saveWorker(), or save()
	// Allow previous runs to be saved to a file system or database so the results
	// can be looked up.

	private static UUID requireId(HttpRequest req, HttpResponse res) throws RequiredException {
		try {
			return UUID.fromString(req.requireQueryParameter("id", res));
		} catch(IllegalArgumentException ex) {  // invalid UUID format
			return null;
		}
	}

	private static MSDWorker requireAction(IOFunction<UUID, MSDWorker> action, UUID id, HttpResponse res) throws RequiredException, IOException {
		MSDWorker msd = action.apply(id);
		if (msd == null) {
			res.status = HttpStatus.NOT_FOUND;
			res.setBody("No MSD Worker with id=" + id);
			HttpHeader.CONTENT_TYPE.to("text/plain; utf-8", res.headers);
			throw new RequiredException();
		}
		return msd;
	}

	@Override
	public void close() throws IOException {
		for (UUID id : new ArrayList<>(workers.keySet()))
			destroyWorker(id);
	}
}
