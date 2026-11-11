# Brief case: storage and first-request latency

One reported paired experiment compared byte-identical Engine files on two
storage locations. Page Cache residency checks supported the intended cold-read
condition. File-read and aggregate initialization times differed, while the
separately measured deserialization interval was similar.

The first-request interval improved, but only one end-to-end pair was available.
The result supports a bounded startup/first-use observation, not a GPU-kernel
or warmed-throughput gain. Raw paths, Engine artifacts, and measurements remain
in the source project; the conclusion should be refreshed with repeated paired
startup and steady-state samples before broader claims.
