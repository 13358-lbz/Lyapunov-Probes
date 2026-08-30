from pathlib import Path

import torch
import typer
from torch import nn


class MLP2(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """Forward pass"""
        return self.layers(x)

    # def score(self, o, hidden_layer_idx):
    #     return torch.sigmoid(
    #         self.layers(
    #             o.hidden_states[hidden_layer_idx][:, -1, :]
    #             .to(self.layers[0].weight.dtype)
    #             .to(self.layers[0].weight.device)
    #         ).squeeze(1)
    #     )
    
    # def score(self, o, hidden_layer_idx):
    #     hidden = o.hidden_states[hidden_layer_idx][:, -1, :]
    #     delta = torch.zeros(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
    #     inputs_with_delta = torch.cat([hidden, delta], dim=1)
    #     # 保证输入和模型参数 dtype 一致
    #     inputs_with_delta = inputs_with_delta.to(self.layers[0].weight.dtype)
    #     return torch.sigmoid(self.layers(inputs_with_delta).squeeze(1))
    
    def score(self, o, hidden_layer_idx):
        hidden = o.hidden_states[hidden_layer_idx][:, -1, :]
        if hidden.shape[1] == 4544:
            delta = torch.zeros(hidden.shape[0], 1, device=hidden.device, dtype=hidden.dtype)
            inputs_with_delta = torch.cat([hidden, delta], dim=1)
        elif hidden.shape[1] == 4545:
            inputs_with_delta = hidden
        else:
            raise ValueError(f"Unexpected hidden size: {hidden.shape[1]}")
        # 保证输入和模型参数 dtype 一致
        inputs_with_delta = inputs_with_delta.to(self.layers[0].weight.dtype)
        return torch.sigmoid(self.layers(inputs_with_delta).squeeze(1))


def preprocess_dataset(dataset):
    """预处理数据集，将is_factual统一转换为float类型"""
    for item in dataset:
        is_factual = item['is_factual']
        if isinstance(is_factual, torch.Tensor):
            if is_factual.dtype == torch.bool:
                item['is_factual'] = is_factual.float()
            elif is_factual.dtype == torch.float16:
                item['is_factual'] = is_factual.float()
        elif isinstance(is_factual, bool):
            item['is_factual'] = torch.tensor(float(is_factual))
        elif isinstance(is_factual, (int, float)):
            item['is_factual'] = torch.tensor(float(is_factual))
    return dataset

def main(
    hidden_path: str = "models/tiiuae/falcon-7b-instruct/lama3_hlayer_0804_24.pt",
    out_name: str = "models/lama_hidden_scorer/tiiuae/falcon-7b-instruct/lama3_hscorer_24_test.pt",
    seed: int = 42,
):
    # Set fixed random number seed
    torch.manual_seed(seed)
    Path(out_name).parent.mkdir(exist_ok=True, parents=True)
    # Prepare dataset
    dataset = torch.load(hidden_path, map_location="cpu")
    
    # 预处理数据集
    dataset = preprocess_dataset(dataset)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    # Initialize the MLP
    # mlp = MLP2(hidden_size=dataset[0]["hidden"].shape[-1])

    # 获取原始特征维度，加1是因为我们要拼接delta（1维）
    original_hidden_size = dataset[0]["hidden"].shape[-1]
    input_size = original_hidden_size + 1  # 关键修改：考虑delta的维度
    # 初始化MLP时使用新的输入维度
    mlp = MLP2(hidden_size=input_size)

    # Define the loss function and optimizer
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # Run the training loop
    for epoch in range(0, 10):  # 5 epochs at maximum
        # Print epoch
        print(f"Starting epoch {epoch}")

        # Set current loss value
        current_loss = 0.0

        # 在训练循环中计算导数惩罚
        for i, data in enumerate(trainloader):
            inputs = data["hidden"].float()
            targets = data["is_factual"].float().unsqueeze(1)

            # 扰动大小（形状：(batch_size, 1)），对于targets不为0的部分，原本的扰动（delta）越大，targets越小 delta 的大小为1-targets，但是为0 的扰动大小为0
            delta = torch.where(targets == 0, torch.tensor(0.0, device=targets.device), 1 - targets).requires_grad_(True)

            # 修正targets为0或1，除了本身为0的其他都修正为1
            targets = torch.where(targets == 0, torch.tensor(0.0, device=targets.device), torch.tensor(1.0, device=targets.device))
            
            # 前向传播（保留计算图用于求导）
            inputs_with_delta = torch.cat([inputs, delta], dim=1)  # 假设模型输入包含δ
            outputs = mlp(inputs_with_delta)
            y_pred = torch.sigmoid(outputs)
            
            # 原始BCE损失
            bce_loss = loss_function(outputs, targets)
            
            # 计算mlp输出对δ的导数（检查单调性）
            mlp_derivatives = torch.autograd.grad(
                outputs.sum(),  # 对输出之和求导（简化计算）
                delta, 
                create_graph=True  # 保留二阶导数用于反向传播
            )[0]
            
            # 惩罚导数为正的情况（违反单调性）
            deriv_penalty = torch.mean(torch.relu(mlp_derivatives))  # relu(x) = max(0, x)
            
            # 总损失
            lambda_penalty = 1e-4  # 惩罚系数
            total_loss = bce_loss + lambda_penalty * deriv_penalty
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Print statistics
            current_loss += total_loss.item()
            if i % 100 == 0:
                print("Loss after mini-batch %5d: %.3f" % (i, current_loss / 500))
                print(
                    "accuracy: ",
                    torch.nn.functional.sigmoid(outputs)
                    .round()
                    .eq(targets)
                    .sum()
                    .item()
                    / len(targets),
                )
                print("mlp.layers[0].weight.sum(): ", mlp.layers[0].weight.sum())
                current_loss = 0.0
        # save model
        torch.save(mlp, out_name)

    # Process is complete.
    print("Training process has finished.")

    # Process is complete.
    print("Training process has finished.")


if __name__ == "__main__":
    typer.run(main)
