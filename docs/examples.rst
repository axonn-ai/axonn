
********
Examples
********

Training
============



Fine-tuning
===========

This `example <https://github.com/axonn-ai/axonn-examples/blob/develop/llm_finetuning/run_clm_no_trainer.py>`_ shows how to fine-tune a causal langauge model using AxoNN as a backend to accelerate. 

To use AxoNN with the Hugging Face Accelerate API, you need to declare the AxoNN plugin and specify the required arguments:

.. code-block:: python 

    from accelerate import AxoNNPlugin
    axonn_plugin = AxoNNPlugin(G_intra_depth=args.tensor_parallelism, G_intra_col=1, G_intra_row=1)
    accelerator = Accelerator(..., axonn_plugin=axonn_plugin, ...)

In the above code block, `G_intra_depth` is, `G_intra_col` is, and `G_intra_row` is

In addition, our context manager is used to parallelize linear layers within the model architecture. Essentially, this is performing the monkey patching mentioned in the EasyAPI section within the User Guide automatically. This enables AxoNN to fine-tune the given model in parallel.

.. code-block:: python 

    with axonn.models.transformers.parallelize(args.model_name_or_path):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                trust_remote_code=args.trust_remote_code,
            )

Inference
=========
