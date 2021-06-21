        split_forward = []
        split_backward = []
        diff_list = []
        diff_add = torch.zeros(x3_reshape.shape, device="cuda")
        for i in range(1, self.n_segment):
            split_forward.append(x3_reshape.split([self.n_segment - i, i], dim=1)[0])
            split_backward.append(x3_conv_reshape.split([i, self.n_segment - i], dim=1)[1])
            diff = split_backward[i - 1] - split_forward[i - 1]
            diff_pad = F.pad(diff, (0, 0, 0, 0, 0, 0, 0, i), mode='constant', value=0)
            diff_list.append(diff_pad)
            diff_add += diff_pad
            print(f'diff[{i - 1}].shape ==>  {diff_list[i - 1].shape}')
        print(f'diff_add.shape ==> {diff_add.shape}')
