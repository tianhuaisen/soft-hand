import matplotlib.pyplot as plt
import math

def result_visualization(loss_list: list, correct_on_test: list, correct_on_train: list,
                         test_interval: int, d_model: int, q: int,
                         v: int, h: int, N: int, dropout: float,
                         DATA_LEN: int, BATCH_SIZE: int, time_cost: float,
                         EPOCH: int, draw_key: int,reslut_figure_path: str,
                         optimizer_name: str,file_name: str, LR: float):

    plt.style.use('seaborn')

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(313)

    ax1.plot(loss_list)
    ax2.plot(correct_on_test, color='red', label='on Test Dataset')
    ax2.plot(correct_on_train, color='blue', label='on Train Dataset')


    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2.set_xlabel(f'epoch/{5}')
    ax2.set_ylabel('correct')
    ax1.set_title('LOSS')
    ax2.set_title('CORRECT')

    plt.legend(loc='best')


    fig.text(x=0.13, y=0.45, s= f'best loss:{min(loss_list)}''   '
                              f'best loss epoch:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}''   '
                              f'last loss:{loss_list[-1]}''\n'
                              f'best correct:test_data:{max(correct_on_test)}% train_data:{max(correct_on_train)}%''    '
                              f'best correct epoch:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}''    '
                              f'last correct:{correct_on_test[-1]}%''\n'
                              f'd_model={d_model} q={q} v={v} h={h} N={N} drop_out={dropout}''  'f'time cost{round(time_cost, 2)}min')

    # Keep result picture
    if EPOCH >= draw_key:
        plt.savefig(
            f'{reslut_figure_path}/{file_name} {max(correct_on_test)}% {optimizer_name} epoch={EPOCH} batch={BATCH_SIZE} lr={LR} [{d_model},{q},{v},{h},{N},{dropout}].png')

    plt.show()

    print('正确率列表', correct_on_test)

    print(f'最小loss：{min(loss_list)}\r\n'
          f'最小loss对应的epoch数:{math.ceil((loss_list.index(min(loss_list)) + 1) / math.ceil((DATA_LEN / BATCH_SIZE)))}\r\n'
          f'最后一轮loss:{loss_list[-1]}\r\n')

    print(f'最大correct：测试集:{max(correct_on_test)}\t 训练集:{max(correct_on_train)}\r\n'
          f'最correct对应的已训练epoch数:{(correct_on_test.index(max(correct_on_test)) + 1) * test_interval}\r\n'
          f'最后一轮correct:{correct_on_test[-1]}')

    print(f'共耗时{round(time_cost, 2)}分钟')