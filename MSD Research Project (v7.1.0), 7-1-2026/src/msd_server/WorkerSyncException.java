package msd_server;

import java.io.IOException;

/**
 * Used to represent that an {@link MSDWorker} has desynced with it's underlying
 * python worker.
 */
public class WorkerSyncException extends IOException {
	public WorkerSyncException(String message) {
		super(message);
	}

	public WorkerSyncException() {
		super();
	}
}