import matplotlib.pyplot as plt

def filetodata(file, column):
    value = []
    with open(file, "r") as f:
        for line in f:
            data = line.split(",")
            if "100%" in data[0]:
                value.append(float(data[column].split(":")[1]))
    return value

def filetotestdata(file, column):
    value = []
    with open(file, "r") as f:
        for line in f:
            data = line.split(",")
            if "Test " in data[0]:
                value.append(float(data[column].split(":")[1]))
    return value

def plot(values, epochs, labels, title, ylabel):
    x = range(epochs)
    fig = plt.figure()
    for i, value in enumerate(values):
        plt.plot(x, value, label = labels[i])
    plt.legend(loc = "best")
    fig.suptitle(title, fontsize=20)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(ylabel, fontsize=16)
    filename = "plots/" + title + ".jpg"
    plt.show()
    fig.savefig(filename)

relu01 = "log/student_relu.log"
relu001 = "log/student_relu001.log"
labels = ["lr = 0.1", "lr = 0.01"]
#
# lossvalues = [filetodata(relu01, 1), filetodata(relu001, 1)]
# plot(lossvalues, 32, labels, "Student with ReLu Loss Training Results", "loss")
# top1values = [filetodata(relu01, 2), filetodata(relu001, 2)]
# plot(top1values, 32, labels, "Student with ReLu Top1 Training Results", "accuracy")
# top5values = [filetodata(relu01, 3), filetodata(relu001, 3)]
# plot(top5values, 32, labels, "Student with ReLu Top5 Training Results", "accuracy")
#
# top1values = [filetotestdata(relu01, 1), filetotestdata(relu001, 1)]
# plot(top1values, 32, labels, "Student with ReLu Top1 Testing Results", "accuracy")
# top5values = [filetotestdata(relu01, 2), filetotestdata(relu001, 2)]
# plot(top5values, 32, labels, "Student with ReLu Top5 Testing Results", "accuracy")
#
swish01 = "log/student_swish01.log"
swish001 = "log/student_swish.log"
# labels = ["lr = 0.1", "lr = 0.01"]
#
# lossvalues = [filetodata(swish01, 1), filetodata(swish001, 1)]
# plot(lossvalues, 35, labels, "Student with Swish Loss Training Results", "loss")
# top1values = [filetodata(swish01, 2), filetodata(swish001, 2)]
# plot(top1values, 35, labels, "Student with Swish Top1 Training Results", "accuracy")
# top5values = [filetodata(swish01, 3), filetodata(swish001, 3)]
# plot(top5values, 35, labels, "Student with Swish Top5 Training Results", "accuracy")

# top1values = [filetotestdata(swish01, 1), filetotestdata(swish001, 1)]
# plot(top1values, 35, labels, "Student with Swish Top1 Testing Results", "accuracy")
# top5values = [filetotestdata(swish01, 2), filetotestdata(swish001, 2)]
# plot(top5values, 35, labels, "Student with Swish Top5 Testing Results", "accuracy")

resnet01 = "log/resnet01.log"
# labels = ["lr = 0.1", "lr = 0.01"]
#
# lossvalues = [filetodata(resnet01, 1)]
# plot(lossvalues, 81, labels, "Teacher with Swish Loss Training Results", "loss")
# top1values = [filetodata(resnet01, 2)]
# plot(top1values, 81, resnet01, "Teacher with Swish Top1 Training Results", "accuracy")
# top5values = [filetodata(resnet01, 3)]
# plot(top5values, 81, labels, "Teacher with Swish Top5 Training Results", "accuracy")

# top1values = [filetotestdata(resnet01, 1)]
# plot(top1values, 81, resnet01, "Teacher with Swish Top1 Testing Results", "accuracy")
# top5values = [filetotestdata(resnet01, 2)]
# plot(top5values, 81, labels, "Teacher with Swish Top5 Testing Results", "accuracy")

kd015 = "log/kd_01_5.log"
kd0110 = "log/kd_01_10.log"
kd055 = "log/kd_05_5.log"
kd0510 = "log/kd_05_10.log"
kd0910 = "log/kd_09_10.log"

labels = ["a=0.1, t=5", "a=0.1, t=10", "a=0.5, t=5", "a=0.5, t=10", "a=0.9, t=10"]

# lossvalues = [filetodata(kd015, 1), filetodata(kd0110, 1), filetodata(kd055, 1), filetodata(kd0510, 1), filetodata(kd0910, 1)]
# plot(lossvalues, 35, labels, "Student with KD Loss Training Results", "loss")
# top1values = [filetodata(kd015, 2), filetodata(kd0110, 2), filetodata(kd055, 2), filetodata(kd0510, 2), filetodata(kd0910, 2)]
# plot(top1values, 35, resnet01, "Student with KD Top1 Training Results", "accuracy")
# top5values = [filetodata(kd015, 3), filetodata(kd0110, 3), filetodata(kd055, 3), filetodata(kd0510, 3), filetodata(kd0910, 3)]
# plot(top5values, 35, labels, "Student with KD Top5 Training Results", "accuracy")

top1values = [filetotestdata(kd015, 1), filetotestdata(kd0110, 1), filetotestdata(kd055, 1), filetotestdata(kd0510, 1), filetotestdata(kd0910, 1)]
plot(top1values, 35, resnet01, "Student with KD Top1 Testing Results", "accuracy")
top5values = [filetotestdata(kd015, 2), filetotestdata(kd0110, 2), filetotestdata(kd055, 2), filetotestdata(kd0510, 2), filetotestdata(kd0910, 2)]
plot(top5values, 35, labels, "Student with KD Top5 Testing Results", "accuracy")
