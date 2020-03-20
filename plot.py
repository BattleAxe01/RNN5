import matplotlib.pyplot as plt


def plot(x, y_pred, y_real):
    plt.plot(x, y_pred, color="blue", label="Predcticted Price")
    plt.plot(x, y_real, color="red", label="Real Price")

    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock")
    plt.legend()
    plt.show()
