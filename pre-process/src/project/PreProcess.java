package project;

import com.codepoetics.protonpack.StreamUtils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class PreProcess {

    public static <T> List<T> mapAndCollect(Map<Long, List<BaseRecord>> customerRecords, Function<List<BaseRecord>, List<T>> mapFunc) {
        return customerRecords
                .keySet()
                .stream()
                .map(custId -> mapFunc.apply(customerRecords.get(custId)))
                .flatMap(Collection::stream)
                .collect(Collectors.toList());
    }

    public static void main(String[] args) throws IOException {

        BaseRecord[] records = Files.readAllLines(Paths.get("train.csv"))
                .stream()
                .filter(line -> !line.startsWith("customer_ID"))
                .map(line -> line.split(","))
                .map(BaseRecord::new)
                .toArray(BaseRecord[]::new);

        Map<Long, List<BaseRecord>> customerRecords = Arrays.stream(records).collect(Collectors.groupingBy(record -> record.customer_ID));

        List<IsContiguousRecord> contiguousRecords = mapAndCollect(customerRecords, (cust) -> {
            Collections.sort(cust);
            int i;
            for (i = 1; i < cust.size(); i++) {
                if (Math.abs(ChronoUnit.MINUTES.between(cust.get(i - 1).time, cust.get(i).time)) > IsContiguousRecord.CONTIGUOUS_TIME) {
                    break;
                }
            }
            final boolean isContiguous = i == cust.size();
            return cust.stream().filter(r -> r.record_type == 1).map(r -> new IsContiguousRecord(r, isContiguous)).collect(Collectors.toList());
        });

        List<InitialPlanRecord> initialPlanRecords = mapAndCollect(customerRecords, (cust) -> {
            Collections.sort(cust);
            BaseRecord first = cust.get(0);
            BaseRecord prev = cust.get(cust.size() - 2);
            long sum_a = (long) (cust.stream().mapToDouble(c -> c.A).sum() / cust.size());
            long sum_b = (long) (cust.stream().mapToDouble(c -> c.B).sum() / cust.size());
            long sum_c = (long) (cust.stream().mapToDouble(c -> c.C).sum() / cust.size());
            long sum_d = (long) (cust.stream().mapToDouble(c -> c.D).sum() / cust.size());
            long sum_e = (long) (cust.stream().mapToDouble(c -> c.E).sum() / cust.size());
            long sum_f = (long) (cust.stream().mapToDouble(c -> c.F).sum() / cust.size());
            long sum_g = (long) (cust.stream().mapToDouble(c -> c.G).sum() / cust.size());
            return cust.stream().filter(r -> r.record_type == 1).map(r -> new InitialPlanRecord(r, first.A, first.B, first.C, first.D, first.E, first.F, first.G, sum_a, sum_b, sum_c, sum_d, sum_e, sum_f, sum_g, prev.A, prev.B, prev.D, prev.E, prev.F, prev.G)).collect(Collectors.toList());
        });

        List<ProcessedRecord> s = StreamUtils.zip(initialPlanRecords.stream(), contiguousRecords.stream(), (ir, cr) -> new ProcessedRecord(ir.baseRecord.getCompleteRecord(), ir, cr)).filter(r -> r.baseRecord.record_type == 1).collect(Collectors.toList());
        printRecords(s::stream, "complete");

        s = StreamUtils.zip(initialPlanRecords.stream(), contiguousRecords.stream(), (ir, cr) -> new ProcessedRecord(ir.baseRecord.getSquashedRecord(), ir, cr)).filter(r -> r.baseRecord.record_type == 1).collect(Collectors.toList());
        printRecords(s::stream, "complete_squashed");

        for (char i = 'A'; i <= 'G'; i++) {
            final char c = i;
            s = StreamUtils.zip(initialPlanRecords.stream(), contiguousRecords.stream(), (ir, cr) -> new ProcessedRecord(ir.baseRecord.getRecord(Character.toString(c)), ir, cr)).collect(Collectors.toList());
            printRecords(s::stream, "complete_" + c);
        }
    }

    public static void printRecords(Supplier<Stream<? extends CSV<?>>> records, String name) throws IOException {
        StringBuilder data = new StringBuilder();
        StringBuilder dr = new StringBuilder(records.get().findFirst().get().header()).append('\n');
        StringBuilder arff = new StringBuilder(records.get().findFirst().get().arffHeader());

        Map<String, Double> max = new HashMap<>();
        Map<String, Double> min = new HashMap<>();

        records.get().forEach(r -> StreamUtils.zip(r.fields(), r.values().mapToDouble(l -> (double) l).boxed(), (f, v) -> {
            double ma = Optional.ofNullable(max.get(f)).orElse(Double.MIN_VALUE);
            if (v > ma) {
                max.put(f, v);
            }
            double mi = Optional.ofNullable(min.get(f)).orElse(Double.MAX_VALUE);
            if (v < mi) {
                min.put(f, v);
            }
            return "";
        }).count());

        List<String> skipNormalize = Arrays.asList("A", "B", "C", "D", "E", "F", "G", "ABCDEFG");

        records.get()
                .limit(500)
                .map(r -> StreamUtils.zip(r.fields(), r.values().mapToDouble(l -> (double) l).boxed(), (f, v) -> skipNormalize.contains(f) ? v : (v - min.get(f)) / (max.get(f) - min.get(f))))
                .forEach(s -> data.append(s.map(Object::toString).collect(Collectors.joining(","))).append('\n'));

        dr.append(data.toString());
        arff.append(data.toString());

        Files.write(Paths.get(name + ".csv"), dr.toString().getBytes());
        Files.write(Paths.get(name + ".arff"), arff.toString().getBytes());

    }

}
