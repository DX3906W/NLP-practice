export BERT_BASE_DIR=./roberta_zh_large
export MY_DATA_DIR=./data
python run_classifier.py
  --task_name=sentimetn_blog \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_large.json \
  --init_checkpoint=$BERT_BASE_DIR/roberta_zh_large_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=6 \
  --learning_rate=2e-5 \
  --num_train_epochs=4.0 \
  --output_dir=./model
