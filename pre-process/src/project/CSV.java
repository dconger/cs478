package project;

import java.util.stream.Stream;

/**
 *
 */
public interface CSV<T> extends Comparable<T> {
    String header();
    default String arffHeader() {
        StringBuilder header = new StringBuilder("@RELATION name\n");
        fields().forEach(f -> header.append("@ATTRIBUTE ").append(f).append(" REAL").append('\n'));
        header.append("@DATA\n");
        return header.toString();
    }
    Stream<String> fields();
    Stream<Long> values();
}
