## LLaMA Summarization

This project uses Meta's LLaMA model to perform text summarization. The script runs a pretrained LLaMA model to generate summaries based on the provided test data.

### Usage

To run the summarization script, use the following command:
```
python                  summarization_LlaMA.py      \
--huggingface_token     yourtoken                   \
--model_name_or_path    meta-llama/Meta-Llama-3-8B  \
--test_file             ./test_data-10.json         \
--max_length            2048                        \
--truncation            True                        \
--return_tensors        pt                          \
--max_new_tokens        512                         \
--do_sample             True                        \
--top_k                 50                          \
--top_p                 0.92                        \
--temperature           0.7                         \
--num_beams             1                           \
--output_dir            ./summarization_LlaMA3.json
```
Replace yourtoken with your Hugging Face token.

#### Arguments:

- **`huggingface_token`**: Your Hugging Face API token to access the pretrained models.
- **`model_name_or_path`**: The path or name of the LLaMA model. In this example, it's using `meta-llama/Meta-Llama-3-8B`.
- **`test_file`**: Path to the test data file, which contains the text you want to summarize.
- **`max_length`**: The maximum input length for each text.
- **`truncation`**: If `True`, truncate sequences longer than `max_length`.
- **`return_tensors`**: Return the tensors in PyTorch format (`pt`).
- **`max_new_tokens`**: The maximum number of tokens to generate.
- **`do_sample`**: Whether to sample the output randomly instead of greedy decoding.
- **`top_k`**: The number of highest probability vocabulary tokens to keep for sampling.
- **`top_p`**: The cumulative probability threshold for nucleus sampling.
- **`temperature`**: Sampling temperature. Lower values make the output more focused.
- **`num_beams`**: The number of beams for beam search. Set to 1 to disable beam search.
- **`output_dir`**: Path to save the output summarizations.