class NodeEncoder:
    def __init__(self, num_regions: int):
        self.num_regions = num_regions

    def one_hot_encode(self, region_index: int) -> list:
        """Perform one-hot encoding for a given brain region index."""
        if region_index < 0 or region_index >= self.num_regions:
            raise ValueError("Region index out of bounds.")
        return [1 if i == region_index else 0 for i in range(self.num_regions)]

    def encode_all(self) -> list:
        """Generate one-hot encodings for all brain regions."""
        return [self.one_hot_encode(i) for i in range(self.num_regions)]