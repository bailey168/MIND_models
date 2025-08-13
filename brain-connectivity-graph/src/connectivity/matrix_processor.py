class MatrixProcessor:
    def __init__(self, matrix):
        if matrix.shape != (21, 21):
            raise ValueError("Matrix must be 21x21.")
        self.matrix = matrix

    def extract_upper_triangle(self):
        upper_triangle_indices = np.triu_indices(21, k=1)
        upper_triangle_values = self.matrix[upper_triangle_indices]
        return upper_triangle_indices, upper_triangle_values

    def get_top_edges(self, upper_triangle_values):
        threshold_index = round(len(upper_triangle_values) * 0.5)
        top_indices = np.argsort(upper_triangle_values)[-threshold_index:]
        return top_indices

    def process_matrix(self):
        upper_triangle_indices, upper_triangle_values = self.extract_upper_triangle()
        top_edge_indices = self.get_top_edges(upper_triangle_values)
        return upper_triangle_indices[0][top_edge_indices], upper_triangle_indices[1][top_edge_indices], upper_triangle_values[top_edge_indices]