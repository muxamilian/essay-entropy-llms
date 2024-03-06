# https://muxamilian.github.io/essay-entropy-llms/

# How to reproduce

To reproduce the result for the *BAWE* dataset and *Mistral-7B*, run the following command:

    python3 analyze.py --raw_input_path final_bawe.json --raw_data_path final_output_bawe_mistral-7B.jsonl

To generate a plot including all results, run:

    python3 plot_all.py
