# imprts
import sys
import getopt

# files
import data
import train
import pred
import plot

all_args = sys.argv[1:]

short_opt = "htp"
long_opt = ["help", "train", "predict"]

args, v = getopt.getopt(all_args, short_opt, long_opt)


# help function
def show_help():
    for i in range(len(short_opt)):
        print(" -" + short_opt[i] + "    --" + long_opt[i])


for current_argument, current_value in args:
    if current_argument in ("-h", "--help"):
        show_help()
    elif current_argument in ("-t", "--train"):
        train.train(data.x_train, data.y_train)
    elif current_argument in ("-p", "--predict"):
        prediction = pred.pred(data.x_test)
        prediction = data.sc.inverse_transform(prediction)
        plot.plot(data.date, prediction, data.y_test)