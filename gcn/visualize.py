from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import time

class Animator:
    def __init__(self):
        """
        Make your boring print() into a beautiful animation

        Usage
        -----
        >>> from animator import Animator
        >>> import numpy as np
        >>> ani = Animator()
        >>> # 设置循环次数
        >>> num_iterations = 20
        >>> 
        >>> for i in range(num_iterations):
        >>>     # 生成一些示例数据，你需要根据你的需求修改这部分
        >>>     x = np.random.rand(10)
        >>>     y = np.random.rand(10)
        >>>     # 绘制散点图
        >>>     ani.ax.scatter(x, y)
        >>>     ani.ax.set_title(f'Iteration {i + 1}')
        >>>     ani.render(0.05) 
        >>> 
        >>> # 关闭图形
        >>> ani.close()
        """
        self.fig, self.ax = plt.subplots()  
    
    def render(self, delay: int = 0.05):
        clear_output(wait=True) # Clear output for dynamic display
        display(self.fig)       # Reset display
        time.sleep(delay)       

    def clear(self):
        self.ax.cla()

    def close(self):
        plt.close(self.fig)


def smooth_loss(loss_list: list, window_size: int = 5):
    """
    平滑损失函数

    Parameters
    ----------
    @param loss_list : list of float
        损失函数列表
    @param window_size : int, optional
        窗口大小, by default 5
    
    Returns
    -------
    smoothed_losses : list of float
        平滑后的损失函数列表
    """
    smoothed_losses = []
    
    for i in range(len(loss_list)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(loss_list), i + window_size // 2 + 1)
        
        # 计算窗口内的平均值
        window_values = loss_list[start_index:end_index]
        smoothed_value = sum(window_values) / len(window_values)
        smoothed_losses.append(smoothed_value)
    
    return smoothed_losses


def show_attention(
        attention,
        xlabel = 'Keys',
        ylabel = 'Queries',
        title = 'Attention weights',
        figsize=(5, 5),
        cmap = 'Reds'
    ):
    """
    画出注意力权重图

    Parameters
    ----------
    @param attention: torch.Tensor
        attention weights, shape (m, n)
    """

    fig = plt.figure(figsize = figsize)

    pcm = plt.imshow(
        attention.detach().numpy(), 
        cmap = cmap
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.colorbar(pcm, shrink=0.7)
    plt.show()