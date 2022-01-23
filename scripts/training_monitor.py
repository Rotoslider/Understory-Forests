import matplotlib.pyplot as plt
import numpy as np


def training_plotter():
    plt.ion()
    fig1 = plt.figure(figsize=(10, 5))
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)


    while 1:
        try:
            ax1.clear()
            ax2.clear()
            ax1.set_xlabel("Num. Epochs")
            ax1.set_ylabel("Loss")
            ax2.set_xlabel("Num. Epochs")
            ax2.set_ylabel("Accuracy")
            training_history = np.loadtxt("../model/training_history.csv")
            plt.suptitle("Training History")
            if training_history.shape[0] > 1:
                ax1.plot(training_history[:, 0], training_history[:, 1])
                ax1.plot(training_history[:, 0], training_history[:, 3])
                ax2.plot(training_history[:, 0], training_history[:, 2])
                ax2.plot(training_history[:, 0], training_history[:, 4])

            plt.draw()
            plt.pause(60)

        except OSError or IndexError:
            pass


if __name__ == "__main__":
    training_plotter()