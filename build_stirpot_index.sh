#!/bin/bash
python -m flashrag.retriever.index_builder \
  --retrieval_method openai_text-embedding-3-small \
  --model_path openai_text-embedding-3-small \
  --corpus_path scratch/stirpot_corpus.jsonl \
  --save_dir scratch/indexes/ \
  --use_fp16 \
  --max_length 512 \
  --batch_size 256 \
  --pooling_method mean \
  --faiss_type Flat