import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 读取CSV文件
df = pd.read_csv('data2024.csv')

# 确保日期列是日期类型
df['Date'] = pd.to_datetime(df['Date'])

# 设置日期列为索引，并确保索引是单调的（有序的）
df.set_index('Date', inplace=True)
df = df.sort_index()

# 定义训练集和测试集的日期范围
train_start_date = '2024-01-02'
train_end_date = '2024-06-28'
test_start_date = '2024-07-01'
test_end_date = '2024-09-27'

# 检查日期范围是否存在于索引中
if (train_start_date in df.index) and (train_end_date in df.index) and (test_start_date in df.index) and (test_end_date in df.index):
    # 根据日期范围划分训练集和测试集
    train = df.loc[train_start_date:train_end_date, 'Price']
    test = df.loc[test_start_date:test_end_date, 'Price']

    # 确定ARIMA模型的参数
    model = auto_arima(train, seasonal=True, suppress_warnings=True, error_action="ignore", trace=True)

    # 拟合ARIMA模型到训练数据
    arima_model = ARIMA(train, order=model.order)
    arima_model_fit = arima_model.fit()

    # 使用模型进行预测
    forecast = arima_model_fit.forecast(steps=len(test))

    # 评估预测结果
    plt.figure(figsize=(12, 6), dpi=1000)
    plt.plot(train.index, train, label='Training Data')
    plt.plot(test.index, test, label='Test Data')
    plt.plot(test.index, forecast, label='Forecast', color='red')
    plt.title('ARIMA Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # 计算预测的准确性
    mse = mean_squared_error(test, forecast)
    print(f"Mean Squared Error: {mse}")
else:
    print("One or more of the specified dates do not exist in the index.")