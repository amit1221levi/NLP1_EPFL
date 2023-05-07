#### `tensorboard/`

In this directory, your tensorboard logs should be stored when you're training/evaluating the models. (as explained below).

For the LSTM variants (three variants in total as explained in the notebook), when inside the training loop and after every the train iteration, **one of the following** should be called (depending on the LSTM variant):

```
tb_writer.add_scalar("LSTM_LM_variant_A/train_loss", train_loss, training_step)
tb_writer.add_scalar("LSTM_LM_variant_B/train_loss", train_loss, training_step)
tb_writer.add_scalar("LSTM_LM_variant_C/train_loss", train_loss, training_step)
```
                          
Moreover, once the training is done and `test_loss` and `test_perplexity` are calculated on the test dataset, the following should be called (only once for each variant). Here we provide the sample code only for variant A, but it should be done similarly for the other two variants.

```
tb_writer.add_scalar("LSTM_LM_variant_A/test_loss", test_loss, 0)
tb_writer.add_scalar("LSTM_LM_variant_A/test_perplexity", test_perplexity, 0)
```

For the seq2seq with attention model (Part 2), the following should be called in every iteration of the training loop:
```
tb_writer.add_scalar("LSTM_seq2seq_attention/train_loss", train_loss, training_step)
tb_writer.add_scalar("LSTM_seq2seq_attention/validation_loss", validation_loss, training_step)
```

_You can find more information on Tensorboard with pytorch [here](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)._