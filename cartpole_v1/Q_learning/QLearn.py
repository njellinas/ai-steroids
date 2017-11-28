from bisect import bisect


class QLearn:
    def quantized(min=0.0, max=1.0, buckets=5):
        if min >= max:
            raise ValueError
        else:
            return [min + k * (max - min) / buckets for k in range(buckets)]

    def matchBucket(value, buckets):
        if value < buckets[0]:
            return None
        elif value > buckets[1] - buckets[0] + buckets[-1]:
            return None
        else:
            idxes = list(range(len(buckets)))
            i = bisect(buckets, value) - 1
            return idxes[i]


if __name__ == '__main__':
    min, max, num_buckets = -5.0, 10.0, 5
    my_buckets = QLearn.quantized(min, max, num_buckets)
    print('min: {}, max: {}, quantized as {} buckets: {}'.format(min, max, num_buckets, my_buckets))
    example_vals = [-5.0, 3.9999, 10.0, 11.0, -5.1]
    for value in example_vals:
        print('value: {} falls in buckets[{}]'.format(value, QLearn.matchBucket(value, my_buckets)))
