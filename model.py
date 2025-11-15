from torch import nn
import torch
import math
import torch.nn.functional as F

"""
Define hyper-parameters here
"""
n_vocab_size = 50257 # vocabulary size
n_context_length = 256 # context length of model, look at n steps and predict n+1 th step
n_embedding = 384 # size of embedding vector => n_head * n_head_size
n_heads = 8 # number of self-attention heads
n_head_size = 64 # number of head size or Q,K & V head size
dropout_rate = 0.33 # dropout rate to avoid over-fitting
n_layers = 6 # number of (multi-head + feed forward) blocks
device = 'cuda' if torch.cuda.is_available() else 'cpu' # assign GPU if available


class SinusoidalPositions(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(-1) # S, 1
        # inside sine / cosine we have pos * (10_000**-2m/d)
        # for stability, calculate instead exp(-2m/d * log(10_000))
        # multiplier shape D/2, then S, 1 * D/2 -> S, D/2
        multiplier = torch.exp((torch.arange(0, d_model, 2) / d_model) * -math.log(10_000))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * multiplier) # S, D/2
        pe[:, 1::2] = torch.cos(position * multiplier)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape B, S, D
        batch_seq_len = x.shape[1]
        return x + self.pe[:batch_seq_len, :].to(x.device)

"""
RoPE Embedding

"""
class RoPE(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self):
        pass

"""
Class for Single head attention
with query (Q), Key (K) & Value matrices.

** masking to be done for next token prediction
"""
class SingleHeadAttention(nn.Module):
    def __init__(self, n_embedding = n_embedding, n_head_size = n_head_size):
        super().__init__()
        self.head_size = n_head_size
        self.n_embedding = n_embedding
        self.query = nn.Linear(self.n_embedding, self.head_size, bias = False)
        self.key = nn.Linear(self.n_embedding, self.head_size, bias = False)
        self.value = nn.Linear(self.n_embedding, self.head_size, bias = False)
        # triangular matrix to focus only on left sequences - causal mask
        self.register_buffer('left_mask', torch.tril(torch.ones(n_context_length, n_context_length)))
        
        # add dropout later
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x, input_padding_mask = None):
        x = x.to(device)
        B, T, C = x.shape # B - batch size, T - Context length, C - vocab size
        # Query (Q)
        #print("Before Query shape: ", x.shape)
        q = self.query(x)
        #print("Query shape: ", q.shape)
        
        # Key (K)
        k = self.key(x)
        #print("Key shape: ", k.shape)
        
        # value (V)
        v = self.value(x)
        #print("Value shape: ", v.shape)
        
        # calculate attention - part 1: Q.K-transpose / sqrt(head_size) - scale the values
        weight = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        #print("After Q.K_transpose shape: ", x.shape)
        
        # mask next word - causal mask
        weight = weight.masked_fill(self.left_mask[:T, :T] == 0, float("-inf")) # softmax of -inf is 0.
        #print("After Causal mask, shape: ", x.shape)
        
        # mask for padding tokens
        if input_padding_mask is not None:
            #print(input_padding_mask.unsqueeze(1).shape)
            bef_wgt = weight
            weight = weight.masked_fill(input_padding_mask.unsqueeze(1) == 0, float("-inf"))
        
        """
        torch.set_printoptions(profile="full")
        with open(f"./analyse_pad.txt", 'a') as f:
            f.write("input_padding_mask")
            f.write(str(input_padding_mask.shape))
            f.write(str(input_padding_mask))
            f.write("\n")
            f.write("weight")
            f.write(str(weight.shape))
            f.write(str(weight))
            f.write("\n")
            f.write("bef_wgt")
            f.write(str(bef_wgt.shape))
            f.write(str(bef_wgt))
        torch.set_printoptions(profile="default")
        
        # tested with this writeup, and the same seems to mask the before and after for padding tokens
        """
        
        # apply softmax to get probability distribution - softmax(Q.K_transpose/sqrt(head size))
        weight = F.softmax(weight, dim = -1)
        
        # add dropout here before MatMul with V
        weight = self.dropout(weight)
        
        # output of self attention block post MatMul (Matrix multiply) of value
        output = weight @ v
        
        #print("After matmul with Value shape: ", output.shape)
        return output

"""
Multi-head attention using single-head attention
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads = n_heads, head_size = n_head_size, n_embedding = n_embedding):
        super().__init__()
        # build list of self-attention blocks with same head_size
        # provide number of heads to let the model know how many multi-head attention to build
        self.heads = nn.ModuleList([SingleHeadAttention(n_embedding, head_size) for _ in range(num_heads)])
        # down-projection to fit into next feed-forward network
        self.projection_down = nn.Linear(n_embedding, n_embedding)
        
        # add dropout later
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def forward(self, x, input_padding_mask = None):
        out = torch.cat([h(x, input_padding_mask) for h in self.heads], dim = -1)
        #print("After Multi-head concat, the shape: ", out.shape)
        out = self.projection_down(out)
        #print("After projection down: ", out.shape)
        out = self.dropout(out)
        return out

"""
Design for feed forward network
"""
class FeedForwardNetwork(nn.Module):
    def __init__(self, n_embedding = n_embedding):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embedding, n_embedding*4), # layer 1 - inp x inp*4 dimension
            nn.GELU(), # can test with GELU, PReLU for more distributions close to 0, rather than 0 with ReLU
            nn.Linear(n_embedding*4, n_embedding), # layer 2 - inp*4 x inp dimension
            # add dropout later
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.network(x)

"""
Design for multi-head + feed-forward Blocks

** Add residual/skip connections here.
** Add Layer normalization to make the layer mean and s.d. to have a proper gaussian distribution.
"""
class BlocksMhFfn(nn.Module):
    def __init__(self, n_embedding = n_embedding, n_heads = n_heads):
        super().__init__()
        head_size = n_embedding // n_heads
        # multi-head self attention
        self.mh_sa = MultiHeadAttention(n_heads, head_size, n_embedding)
        # feed-forward
        self.f_fwd = FeedForwardNetwork(n_embedding)
        
        # define layer normalization here
        self.ly_nm_1 = nn.LayerNorm(n_embedding)
        self.ly_nm_2 = nn.LayerNorm(n_embedding)
    
    def forward(self, x, input_padding_mask = None):
        # here we add skip/residual connections by adding x with f(x) --> y = x + f(x)
        # also apply layer norm before passing to multi-head attention or feed fwd network.
        x = x + self.mh_sa(self.ly_nm_1(x), input_padding_mask)
        x = x + self.f_fwd(self.ly_nm_2(x))
        #print("After FFn, the shape: ", x.shape)
        return x

"""
Building Sequential class with a forward pass and allow this to accept kwargs,
required for passing in padding mask to single head block.
"""
class KwSequential(nn.Sequential):
    def forward(self, x, pad_mask = None):
        for m in self:
            x = m(x, pad_mask)
        return x

"""
Decoding strategy
~ based on temp, top-p, top-k or beam search strategies
"""
class DecodeStrategy(nn.Module):
    def __init__(self, temperature = 0, top_p = 0, top_k = 0):
        super().__init__()
        pass
    
    def predict(self):
        pass

"""
This block will contain transformer model's
neural architecture.

** vocab_size - optional, set to 50k here, or vocab size passed.
Present architecture:
1. Token embedding + Positional embedding
2. Self-Attention with n-heads

"""
class LanguageModel(nn.Module):
    def __init__(self, n_vocab_size = n_vocab_size):
        super().__init__()
        # define token encoding
        self.token_embedding = nn.Embedding(n_vocab_size, n_embedding)
        # instantiate positional encoding class
        self.position_encode = SinusoidalPositions(n_vocab_size, n_embedding)
        
        # Self-attention ~ Single_head, Multi-head & Block-multi-head # can be changed here. 
        #self.blocks_layer = nn.Sequential(*[BlocksMhFfn(n_embedding, n_heads) for _ in range(n_layers)])
        #self.blocks_layer = SingleHeadAttention(n_embedding, n_head_size)
        # used this block to pass in arguments to forward pass, sequential only allows 1 argument hence.
        self.blocks_layer = KwSequential(*[BlocksMhFfn(n_embedding, n_heads) for _ in range(n_layers)])
        
        
        # layer normalization final layer
        self.ly_norm_fin = nn.LayerNorm(n_embedding)
        # dropout try - if final layer will be further deep
        self.dropout = torch.nn.Dropout(dropout_rate)
        # language model linear layer
        self.linear_lang_layer = nn.Linear(n_embedding, n_vocab_size)
        
        # random Initialization weights using mean 0 and s.d. 0.02
        self.apply(self._init_weights)
        
    # Function that does random weight initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input, input_padding_mask = None):
        
        # get token and position encoding
        x = self.token_embedding(input)
        #print("Token embedding shape: ", x.shape)
        x = self.position_encode.forward(x)
        #print("After Position embedding shape: ", x.shape)
        # self-attention with multiple blocks (comprising of multiple heads)
        x = self.blocks_layer(x, input_padding_mask)
        #print("Post MH-blocks shape: ", x.shape)
        # apply final layer normalization
        x = self.ly_norm_fin(x)
        # linear layer
        #print("Before final linear layer: ", x.shape)
        logits = self.linear_lang_layer(x)
        
        #print("After all layers: ", logits.shape)
        return logits
    
    #generate function ~ not required here * but built for scaling this for later stage to something else
    def generate(self, idx, max_new_tokens):
        idx = idx.to(device)
        for _ in max_new_tokens:
            # focus on last context length steps
            inp_x = idx[:, -n_context_length, :]
            # predict
            logits = self(inp_x)
            # focus on last t+1 th step
            last_logit = logits[:, -1, :]
            # apply softmax
            prob_dist = F.softmax(last_logit, dim = -1)
            # sample from the distribution + or to try argmax means temp = 0
            next_idx = torch.multinomial(prob_dist, num_samples = 1)
            # append to idx + nth step, n =1 here.
            idx = torch.cat((idx, next_idx), dim = 1)
        # return data
        return idx
            
        
        
def get_best_model_definition(vocab_size):
    """
    This is the model that will be used in the evaluation script
    Ensure it matches the .pt file provided there
    """
    model = LanguageModel(vocab_size)
    return model

