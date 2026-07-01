package msd_server;

import java.io.IOException;

public interface IOFunction<T, R> {
	R apply(T t) throws IOException;
}
