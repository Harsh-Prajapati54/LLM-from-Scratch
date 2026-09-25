# configuration for language model 

Gpt_config ={
    "vocab_size" : 50527, # vocubulary size
    "context_length" : 1024,
    "emb_dim" : 768 ,
    "num_heads" : 48,   # orignal 12 
    "num_layer" : 32,   # orignal 12 
    "drop_rate" : 0.1,
    "qkv_bias"  : False
}