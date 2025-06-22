# TopoGAT: Topological Attention for Graph Representation Learning

TopoGAT is a novel graph neural network model that integrates persistent homology into the attention mechanism of GAT, enabling better representation learning through topological insights.

## Features
- Integration of persistent homology via topological embeddings
- Enhanced attention mechanism for GNNs
- Stable and expressive graph learning

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Structure
- `config/config.yaml`: Configuration file.
- `data/dataset.py`: Loads and prepares graph dataset.
- `models/topogat.py`: Model architecture.
- `trainers/trainer.py`: Training loop.
- `evaluators/evaluator.py`: Model evaluation.
- `utils/topology.py`: Topological feature computation using persistent homology.
- `utils/logger.py`: Logging utility.
- `utils/tester.py`: Unit tests.

## Configuration
See `config/config.yaml` to change dataset, learning rate, etc.

## Run Unit Tests
```bash
python -m utils.tester
```

## Train
```bash
python main.py
```


## License
[Apache 2.0 License](LICENSE)
