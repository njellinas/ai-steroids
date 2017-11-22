class QLearn:
    def quantize(min=0.0, max=1.0, buckets=5):
        if min >= max:
            raise ValueError
        else:
            return [k * (max - min) / buckets for k in range(buckets)]