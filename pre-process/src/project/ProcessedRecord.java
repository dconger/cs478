package project;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class ProcessedRecord implements CSV<ProcessedRecord> {

    public final BaseRecord baseRecord;
    public final CSV[] records;

    public ProcessedRecord(BaseRecord baseRecord, CSV... records) {
        this.baseRecord = baseRecord;
        this.records = new CSV[records.length + 1];
        this.records[records.length] = baseRecord;
        System.arraycopy(records, 0, this.records, 0, records.length);
    }

    @Override
    public String header() {
        return fields().collect(Collectors.joining(","));
    }

    @Override
    public String toString() {
        return values().map(Object::toString).collect(Collectors.joining(","));
    }

    @Override
    public Stream<String> fields() {
        return Arrays.stream(records).flatMap(CSV::fields);
    }

    @Override
    public Stream<Long> values() {
        return Arrays.stream(records).flatMap(CSV::values);
    }

    @Override
    public int compareTo(ProcessedRecord o) {
        return baseRecord.compareTo(o.baseRecord);
    }
}
