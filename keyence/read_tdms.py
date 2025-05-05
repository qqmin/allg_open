# pip install npTDMS

from nptdms import TdmsFile
import pandas as pd
import matplotlib.pyplot as plt

# 读取 TDMS 文件
tdms_file = TdmsFile.read("tdms/keyence.tdms")

# 获取所有组名列表
group_names = [group.name for group in tdms_file.groups()]
print("实际组名列表:", group_names)  # 输出如 ['时间(s)', '采集数据']

"""
# 遍历所有组对象
for group_obj in tdms_file.groups():
    group_name = group_obj.name  # 获取组名

    # 遍历该组下的通道
    for channel in group_obj.channels():
        data = channel[:]  # 获取通道数据

        df = pd.DataFrame(data)
        print(f"{channel.name}, shape:{data.shape} 预览:\n{df.head()}")
        df.to_csv(f"tdms/{channel.name}.csv", index=False)
"""


# 提取时间和数据通道
try:
    # 获取时间组和时间通道
    time_group = tdms_file["时间(s)"]
    time_channel = time_group["时间"]
    time_data = time_channel[:]

    # 获取数据组和数据通道
    data_group = tdms_file["采集数据"]
    data_channel = data_group["位移量(mm)"]
    data_values = data_channel[:]

except KeyError as e:
    print(f"错误: 找不到指定的组或通道, 请检查名称是否正确. 错误详情: {e}")
    exit()


# 创建 DataFrame 并保存
# 检查数据长度是否一致
if len(time_data) != len(data_values):
    print(f"警告: 时间数据长度({len(time_data)})与采集数据长度({len(data_values)})不一致!")
    # 取最小长度对齐
    min_length = min(len(time_data), len(data_values))
    time_data = time_data[:min_length]
    data_values = data_values[:min_length]

# 创建 DataFrame
df = pd.DataFrame({
    "时间": time_data,
    "位移": data_values
})

# 保存为 CSV
csv_path = "tdms/output_tdms.csv"
df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用 utf-8-sig 避免中文乱码

# 保存为 Excel
excel_path = "tdms/output_tdms.xlsx"
df.to_excel(excel_path, index=False)

print(f"数据已保存至: {csv_path}")


# 数据可视化
plt.figure(figsize=(12, 6), dpi=150)

# 绘制时间序列图
plt.plot(df["时间"], df["位移"],
         linewidth=1,
         color='steelblue',
         label='Vibration Data')

# 添加图表元素
plt.title("T-D", fontsize=14, pad=20)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Displacement (mm)", fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 自动调整时间轴显示
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5))  # 限制时间轴刻度数量
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))  # 限制时间轴刻度数量

# 保存图片
img_path = "tdms/view_tdms.png"
plt.savefig(img_path, bbox_inches='tight')
print(f"可视化图表已保存至: {img_path}")

plt.show()
