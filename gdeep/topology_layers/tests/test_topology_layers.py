# # %%
# model = SmallSetTransformer()

# x = torch.randint(1, 100, (2 ** 10, 9))
# x = torch.unsqueeze(x, -1).float()
# model(x)
# # %%
# total_params = 0
# for parameter in model.parameters():
#     total_params += parameter.nelement()
# print('trainable parameters:', total_params)