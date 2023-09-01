import torch
import torch.nn as nn


class HRCAModule(nn.Module):
    """
    """

    def __init__(self, class_types: dict,
                 step_channel: int,
                 task_channel: int,
                 triplet_channel: int,
                 hidden_dim: int,
                 use_prior_knowledge: bool,
                 use_rlls_rc: bool):
        """

        :param class_types:
        :param input_dim:
        :param out_dim:
        :param num_head:
        """
        super(HRCAModule, self).__init__()

        self.use_prior_knowledge = use_prior_knowledge
        self.use_rlls_rc = use_rlls_rc
        if self.use_prior_knowledge:
            print("    [HRCAModule]] :: <warning>  (TRUE) USING prior knowledge for HRCA CONNECTIONS !!!")
        else:
            print("    [HRCAModule]] :: <warning> (FALSE) DISABLE HRCA CONNECTIONS with prior knowledge !!!")
        if self.use_rlls_rc:
            print("    [HRCAModule]] :: <warning>  (TRUE) USING use_rlls_rc for HRCA CONNECTIONS  !!!")
        else:
            print("    [HRCAModule]] :: <warning> (FALSE) DISABLE use_rlls_rc for HRCA CONNECTIONS !!!")

        self.num_step = step_channel
        self.num_task = task_channel
        self.num_triplet = triplet_channel

        # define the attention mechanism here...
        self.soft = nn.Softmax(dim=1)
        self.step_ch_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.num_step, out_channels=hidden_dim, kernel_size=1, bias=False)
        )
        self.task_ch_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.num_task, out_channels=hidden_dim, kernel_size=1, bias=False)
        )
        self.triplet_ch_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.num_triplet, out_channels=hidden_dim, kernel_size=1, bias=False)
        )

        # hard coding the step-task-triplet relations for now.
        self.step_vs_task = torch.zeros([7, 16])
        self.step_vs_task[0, 0] = 1
        self.step_vs_task[1, 1:6] = 1
        self.step_vs_task[2, 6:9] = 1
        self.step_vs_task[3, 9:11] = 1
        self.step_vs_task[4, 11:13] = 1
        self.step_vs_task[5, 13:15] = 1
        self.step_vs_task[6, 15] = 1

        self.task_vs_triplet = torch.zeros([16, 39])
        # hard code
        tid = {
            "idle": 0,
            "AA1": 1, "AA2": 2, "AA3": 3, "AA4": 4, "AA5": 5, "AA6": 6,
            "AB1": 7, "AB2": 8, "AB3": 9, "AB4": 10, "AB5": 11, "AB6": 12, "AB7": 13, "AB8": 14, "AB9": 15,
            "AB10": 16,
            "AC1": 17, "AC2": 18, "AC3": 19, "AC4": 20, "AC5": 21, "AC6": 22, "AC7": 23,
            "AD1": 24, "AD2": 25, "AD3": 26,
            "AE1": 27,
            "AF1": 28, "AF2": 29,
            "AG1": 30,
            "AH1": 31,
            "AI1": 32, "AI2": 33, "AI3": 34,
            "AJ1": 35,
            "AK1": 36, "AK2": 37, "AK3": 38
        }
        print("    [HRCA] :: (TRUE) USING TRAINING HRCA CONNECTIONS !!!")

        self.task_vs_triplet[0, 0] = 1
        self.task_vs_triplet[1, tid['AA1']] = 1
        self.task_vs_triplet[1, tid['AB1']] = 1
        self.task_vs_triplet[1, tid['AC1']] = 1
        self.task_vs_triplet[2, tid['AA2']] = 1
        self.task_vs_triplet[2, tid['AB2']] = 1
        self.task_vs_triplet[2, tid['AC2']] = 1
        self.task_vs_triplet[3, tid['AA3']] = 1
        self.task_vs_triplet[3, tid['AB3']] = 1
        self.task_vs_triplet[3, tid['AC3']] = 1
        self.task_vs_triplet[4, tid['AA4']] = 1
        self.task_vs_triplet[4, tid['AB4']] = 1
        self.task_vs_triplet[4, tid['AC4']] = 1
        self.task_vs_triplet[5, tid['AA5']] = 1
        self.task_vs_triplet[5, tid['AB5']] = 1
        self.task_vs_triplet[5, tid['AC5']] = 1
        self.task_vs_triplet[6, tid['AA6']] = 1
        self.task_vs_triplet[6, tid['AB6']] = 1
        self.task_vs_triplet[6, tid['AC6']] = 1
        self.task_vs_triplet[6, tid['AD2']] = 1
        self.task_vs_triplet[7, tid['AD1']] = 1
        self.task_vs_triplet[7, tid['AI1']] = 1
        self.task_vs_triplet[8, tid['AD3']] = 1
        self.task_vs_triplet[8, tid['AI1']] = 1
        self.task_vs_triplet[9, tid['AB7']] = 1
        self.task_vs_triplet[9, tid['AC6']] = 1
        self.task_vs_triplet[9, tid['AF1']] = 1
        self.task_vs_triplet[9, tid['AG1']] = 1
        self.task_vs_triplet[9, tid['AH1']] = 1
        self.task_vs_triplet[10, tid['AC1']] = 1
        self.task_vs_triplet[10, tid['AC5']] = 1
        self.task_vs_triplet[10, tid['AC7']] = 1
        self.task_vs_triplet[10, tid['AB8']] = 1
        self.task_vs_triplet[11, tid['AE1']] = 1
        self.task_vs_triplet[12, tid['AJ1']] = 1
        self.task_vs_triplet[13, tid['AB9']] = 1
        self.task_vs_triplet[14, tid['AB9']] = 1
        self.task_vs_triplet[14, tid['AI1']] = 1
        self.task_vs_triplet[14, tid['AI2']] = 1
        self.task_vs_triplet[14, tid['AK1']] = 1
        self.task_vs_triplet[15, tid['AB7']] = 1
        self.task_vs_triplet[15, tid['AB10']] = 1
        self.task_vs_triplet[15, tid['AI1']] = 1

        self.step_vs_triplet = torch.zeros([7, 39])
        self.step_vs_triplet[0, 0] = 1
        self.step_vs_triplet[1, tid['AA1']] = 1
        self.step_vs_triplet[1, tid['AB1']] = 1
        self.step_vs_triplet[1, tid['AC1']] = 1
        self.step_vs_triplet[1, tid['AA2']] = 1
        self.step_vs_triplet[1, tid['AB2']] = 1
        self.step_vs_triplet[1, tid['AC2']] = 1
        self.step_vs_triplet[1, tid['AA3']] = 1
        self.step_vs_triplet[1, tid['AB3']] = 1
        self.step_vs_triplet[1, tid['AC3']] = 1
        self.step_vs_triplet[1, tid['AA4']] = 1
        self.step_vs_triplet[1, tid['AB4']] = 1
        self.step_vs_triplet[1, tid['AC4']] = 1
        self.step_vs_triplet[1, tid['AA5']] = 1
        self.step_vs_triplet[1, tid['AB5']] = 1
        self.step_vs_triplet[1, tid['AC5']] = 1

        self.step_vs_triplet[2, tid['AA6']] = 1
        self.step_vs_triplet[2, tid['AB6']] = 1
        self.step_vs_triplet[2, tid['AC6']] = 1
        self.step_vs_triplet[2, tid['AD2']] = 1
        self.step_vs_triplet[2, tid['AD1']] = 1
        self.step_vs_triplet[2, tid['AI1']] = 1
        self.step_vs_triplet[2, tid['AD3']] = 1

        self.step_vs_triplet[3, tid['AB7']] = 1
        self.step_vs_triplet[3, tid['AC6']] = 1
        self.step_vs_triplet[3, tid['AF1']] = 1
        self.step_vs_triplet[3, tid['AG1']] = 1
        self.step_vs_triplet[3, tid['AH1']] = 1
        self.step_vs_triplet[3, tid['AC1']] = 1
        self.step_vs_triplet[3, tid['AC5']] = 1
        self.step_vs_triplet[3, tid['AC7']] = 1
        self.step_vs_triplet[3, tid['AB8']] = 1

        self.step_vs_triplet[4, tid['AE1']] = 1
        self.step_vs_triplet[4, tid['AJ1']] = 1

        self.step_vs_triplet[5, tid['AB8']] = 1
        self.step_vs_triplet[5, tid['AB9']] = 1
        self.step_vs_triplet[5, tid['AI1']] = 1
        self.step_vs_triplet[5, tid['AI2']] = 1
        self.step_vs_triplet[5, tid['AK1']] = 1

        self.step_vs_triplet[3, tid['AB7']] = 1
        self.step_vs_triplet[3, tid['AB10']] = 1
        self.step_vs_triplet[3, tid['AI1']] = 1

    def forward(self, input: dict):
        """
        :param input: dict
        :return: dict
        """
        batch_size = input['step'].size(0)
        # B x 1 x NUM_CLASS
        embed_step = input['step'].unsqueeze(-1).transpose(2, 1)
        embed_task = input['task'].unsqueeze(-1).transpose(2, 1)
        embed_triplet = input['triplet'].unsqueeze(-1).transpose(2, 1)
        # B x D x NUM_CLASS
        embed_step = self.step_ch_conv(embed_step)
        embed_task = self.task_ch_conv(embed_task)
        embed_triplet = self.triplet_ch_conv(embed_triplet)

        # all inputs are in B x logits
        range_step = torch.max(input['step'], dim=1)[0] - torch.min(input['step'], dim=1)[0]
        range_task = torch.max(input['task'], dim=1)[0] - torch.min(input['task'], dim=1)[0]
        range_triplet = torch.max(input['triplet'], dim=1)[0] - torch.min(input['triplet'], dim=1)[0]

        # step vs task
        # [BxSxD] * [B,D,T] = [B,S,T]
        step_task_corr = embed_step.transpose(2, 1).matmul(embed_task)
        dk = torch.sqrt(torch.tensor(list(embed_step.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        step_task_corr /= dk
        att_st = self.soft(step_task_corr)  # [B,S,T]
        if self.use_prior_knowledge:
            att_st = att_st * torch.stack([self.step_vs_task for i in range(batch_size)], dim=0).to(att_st.device)
        # [B,S,T] * [B,T,1] = [B,S,1] attention from task to step
        att_wst = att_st.matmul(input['task'].unsqueeze(-1))
        att_wst /= range_step.max()

        # step vs triplet
        # [BxSxD] * [B,D,Tr]
        step_triplet_corr = embed_step.transpose(2, 1).matmul(embed_triplet)
        dk = torch.sqrt(torch.tensor(list(embed_step.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        step_triplet_corr /= dk
        att_str = self.soft(step_triplet_corr)

        if self.use_prior_knowledge:
            att_str = att_str * torch.stack([self.step_vs_triplet for i in range(batch_size)], dim=0).to(att_str.device)
        # [B,S,Tr] * [B,Tr,1] = [B,S,1] attention from triplet to step
        att_wstr = att_str.matmul(input['triplet'].unsqueeze(-1))
        att_wstr /= range_step.max()

        # task vs step
        # [BxTx1] * [B,1,S]
        task_step_corr = embed_task.transpose(2, 1).matmul(embed_step)
        dk = torch.sqrt(torch.tensor(list(embed_task.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        task_step_corr /= dk
        # [B, T, S]
        att_ts = self.soft(task_step_corr)
        if self.use_prior_knowledge:
            att_ts = att_ts * torch.stack([self.step_vs_task.transpose(1, 0) for i in range(batch_size)], dim=0).to(
                att_ts.device)
        # [B,T,S] * [B,S,1] = [B, T, 1]
        att_wts = att_ts.matmul(input['step'].unsqueeze(-1))
        att_wts /= range_task.max()

        # task vs triplet
        # [BxTx1] * [B,1,Tr]
        task_triplet_corr = embed_task.transpose(2, 1).matmul(embed_triplet)
        dk = torch.sqrt(torch.tensor(list(embed_task.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        task_triplet_corr /= dk
        att_ttr = self.soft(task_triplet_corr)
        if self.use_prior_knowledge:
            att_ttr = att_ttr * torch.stack([self.task_vs_triplet for i in range(batch_size)], dim=0).to(att_ttr.device)
        # [B,T,Tr] * [B,Tr,1] = [B, T, 1]

        att_wttr = att_ttr.matmul(input['triplet'].unsqueeze(-1))
        att_wttr /= range_task.max()

        # triplet vs step
        # [BxTrx1] * [B,1,S]
        triplet_step_corr = embed_triplet.transpose(2, 1).matmul(embed_step)
        dk = torch.sqrt(torch.tensor(list(embed_triplet.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        triplet_step_corr /= dk
        att_trs = self.soft(triplet_step_corr)
        if self.use_prior_knowledge:
            att_trs = att_trs * torch.stack([self.step_vs_triplet.transpose(1, 0) for i in range(batch_size)],
                                            dim=0).to(att_trs.device)
        # [B,Tr,S] * [B,S,1] = [B, Tr, 1]
        att_wtrs = att_trs.matmul(input['step'].unsqueeze(-1))
        att_wtrs /= range_triplet.max()

        # triplet vs task
        # [BxTrx1] * [BxTx1]
        triplet_task_corr = embed_triplet.transpose(2, 1).matmul(embed_task)
        dk = torch.sqrt(torch.tensor(list(embed_triplet.shape)[-2], dtype=torch.float32))
        if torch.isnan(dk).sum():
            raise RuntimeError
        triplet_task_corr /= dk
        att_trt = self.soft(triplet_task_corr)
        if self.use_prior_knowledge:
            att_trt = att_trt * torch.stack([self.task_vs_triplet.transpose(1, 0) for i in range(batch_size)],
                                            dim=0).to(att_trt.device)
        # [B,Tr,T] * [B,T,1] = [B, Tr, 1]
        att_wtrt = att_trt.matmul(input['task'].unsqueeze(-1))
        att_wtrt /= range_triplet.max()

        return {
            "step_task": att_wst,
            "step_triplet": att_wstr,
            "task_step": att_wts,
            "task_triplet": att_wttr,
            "triplet_step": att_wtrs,
            "triplet_task": att_wtrt
        }
