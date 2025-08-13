# Brain Connectivity Graph

This project aims to convert a 21x21 symmetric matrix representing functional connectivity of the brain into a graph. The graph retains the top 50% of edges with the highest magnitude and uses one-hot encoding for the brain region identity of each node.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To process the matrix and build the graph, run the main script:

```
python src/main.py
```

This will execute the following steps:
1. Load the 21x21 symmetric matrix from the raw data directory.
2. Process the matrix to extract the upper triangle and calculate the top 50% of edges based on magnitude.
3. Build the graph using the processed matrix.
4. Perform one-hot encoding for the brain region identities of each node.

## Directory Structure

```
brain-connectivity-graph
├── src
│   ├── __init__.py
│   ├── connectivity
│   │   ├── __init__.py
│   │   ├── matrix_processor.py
│   │   └── graph_builder.py
│   ├── encoding
│   │   ├── __init__.py
│   │   └── node_encoder.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── main.py
├── data
│   ├── raw
│   └── processed
├── tests
│   ├── __init__.py
│   ├── test_matrix_processor.py
│   ├── test_graph_builder.py
│   └── test_node_encoder.py
├── requirements.txt
├── setup.py
└── README.md
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.