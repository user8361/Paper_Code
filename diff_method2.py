        # ------------------------ #
        # t feature 第 t 帧特征
        split_forward = []
        split_backward = []
        diff_list = []
        diff_add = torch.zeros(reshape_bottleneck.shape, device="cuda")


        # 求帧差的结果（相邻 与 不相邻）
        for i in range(1, self.n_segment):
            split_forward.append(reshape_bottleneck.split([self.n_segment - i, i], dim=1)[0])
            split_backward.append(reshape_conv_bottleneck.split([i, self.n_segment - i], dim=1)[1])
            diff = split_backward[i - 1] - split_forward[i - 1]
            diff_pad = F.pad(diff, (0, 0, 0, 0, 0, 0, 0, i), mode='constant', value=0)
            diff_list.append(diff_pad)
            diff_add += diff_pad # 得到每一列特征差相加后的结果



        # 假设采样 8 帧 ， 需要求得 B1 - B7 ，7列帧差
        diff_backward = []  # 对应 B1 - B7 （下标：0 - 6）
        for i in range(1, self.n_segment):
            diff_backward.append(diff_add[:, i - 1:i, :, :, :])
        # 需要求得 F2 -  F8 （对应下标 0 - 6）
        diff_forward = []
        for i in range(len(diff_list)):
            if i == 0:  # 将 21 添加到列表中
                diff_forward.append(diff_list[i][:, 0:1, :, :, :])
            else:
                # tmp_add = torch.zeros_like(diff_list[0][0])
                for j in range(0, i):
                    tmp_add = diff_list[j][:, i:i + 1, :, :, :]
                    for k in range(0, i):
                        tmp_add += diff_list[j + 1][:, i - 1:i, :, :, :]
                diff_forward.append(tmp_add)  # 斜着相加

        # concat
        diff_concat = diff_backward[0]
        for i in range(0, len(diff_forward)-1):
            tmp = diff_backward[i]+diff_forward[i+1]
            diff_concat = torch.cat((diff_concat,tmp),dim=1)
        diff_concat = torch.cat((diff_concat,diff_forward[-1]),dim=1)
        # ------------------------ #
