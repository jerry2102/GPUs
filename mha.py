        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(
                embed_dim = 8, num_heads = 8, kdim = 4, vdim = 3, batch_first = True)


    def forward(self, q, k, v):
        attn_output, attn_output_weights = self.attention(q, k, v)
        return attn_output, attn_output_weights


q = torch.randint(0, 10, size = (10, 9, 8), dtype = torch.float32)
k = torch.randint(0, 10, size = (10, 7, 4), dtype = torch.float32)
v = torch.randint(0, 10, size = (10, 7, 3), dtype = torch.float32)

attn = Attention()
attn.eval()

a, aw = attn(q, k, v)
print(a)
print(aw)
args = (q, k, v)
torch.onnx.export(model = attn, args = args,
                  f = "./multi_head_attention.onnx",
                  input_names = ["q", "k", "v"],
                  output_names = ["attn", "attn_weight"],
                  opset_version = 11,
                  verbose = True)
