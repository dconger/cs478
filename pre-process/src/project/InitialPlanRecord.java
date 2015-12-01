package project;

import java.lang.reflect.Field;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 */
public class InitialPlanRecord implements CSV<InitialPlanRecord> {

    public final BaseRecord baseRecord;
    public final long initial_a, initial_b, initial_c, initial_d, initial_e, initial_f, initial_g;
    public final long sum_a, sum_b, sum_c, sum_d, sum_e, sum_f, sum_g;
    public final long prev_a, prev_b, prev_d, prev_e, prev_f, prev_g;

    public InitialPlanRecord(BaseRecord baseRecord, long initial_a, long initial_b, long initial_c, long initial_d, long initial_e, long initial_f, long initial_g, long sum_a, long sum_b, long sum_c, long sum_d, long sum_e, long sum_f, long sum_g, long prev_a, long prev_b, long prev_d, long prev_e, long prev_f, long prev_g) {
        this.baseRecord = baseRecord;
        this.initial_a = initial_a;
        this.initial_b = initial_b;
        this.initial_c = initial_c;
        this.initial_d = initial_d;
        this.initial_e = initial_e;
        this.initial_f = initial_f;
        this.initial_g = initial_g;
        this.sum_a = sum_a;
        this.sum_b = sum_b;
        this.sum_c = sum_c;
        this.sum_d = sum_d;
        this.sum_e = sum_e;
        this.sum_f = sum_f;
        this.sum_g = sum_g;
        this.prev_a = prev_a;
        this.prev_b = prev_b;
        this.prev_d = prev_d;
        this.prev_e = prev_e;
        this.prev_f = prev_f;
        this.prev_g = prev_g;
    }

    @Override
    public Stream<String> fields() {
        return Arrays.stream(InitialPlanRecord.class.getFields())
                .map(Field::getName)
                .filter(s -> !"baseRecord".equals(s));
    }

    @Override
    public Stream<Long> values() {
        return Arrays.stream(InitialPlanRecord.class.getFields())
                .filter(field -> !field.getName().equals("baseRecord"))
                .map(field -> BaseRecord.fieldToString(InitialPlanRecord.this, field));
    }

    @Override
    public String header() {
        return baseRecord.header() + "," + fields().collect(Collectors.joining(","));
    }

    @Override
    public String toString() {
        return baseRecord.values() + "," + values().map(Object::toString).collect(Collectors.joining(","));
    }

    @Override
    public int compareTo(InitialPlanRecord o) {
        return baseRecord.compareTo(o.baseRecord);
    }
}
