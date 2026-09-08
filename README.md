# README

[![codecov](https://codecov.io/gh/rithikkulkarni/ask-sklearn/graph/badge.svg)](https://codecov.io/gh/rithikkulkarni/ask-sklearn)

## Evaluation Harness
To run the evaluation harness locally, first start Docker:
```
docker run -p 6333:6333 qdrant/qdrant
```
Then, for smoke (subset of 6 hand-picked validation questions):
```
python -m src.eval.run_eval --smoke
```
And for full validation set (35 questions):
```
python -m src.eval.run_eval --full
```