import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

accuracies = np.load("full_accuracies_slow_multi_mnist.npy")
latent_dims = list(range(1,15+1))
model_names = ["DiME", "CCA", "CKA", "Fully Supervised"]
experiment_name = "multi_mnist"
comparison_type = "mean"

print(accuracies.shape)
print(accuracies)
if comparison_type == "max":
    accuracies = np.amax(accuracies, axis=0)

    latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)

    formattedComps = np.hstack((latent_col, accuracies.T))
    df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
    print(df.to_latex(index=False))
    plt.figure()
    df.plot(x="Latent Dim")
    plt.title("Testing Accuracy vs Latent Dim (max over 10 runs)")

if comparison_type == "min":
    accuracies = np.amin(accuracies, axis=0)

    latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)

    formattedComps = np.hstack((latent_col, accuracies.T))
    df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
    print(df.to_latex(index=False))
    plt.figure()
    df.plot(x="Latent Dim")
    plt.title("Testing Accuracy vs Latent Dim (min)")


elif comparison_type == "mean":
    stds = np.std(accuracies, axis=0)
    means = np.mean(accuracies, axis=0)
    
    print(stds)
    print(means)
    for j in range(means.shape[0]):
        plt.plot(latent_dims, means[j], label=model_names[j])
        plt.fill_between(latent_dims,means[j]-stds[j],means[j]+stds[j],alpha=.1)

    plt.legend()
    plt.title("Supervised Finetuning Classification Accuracy vs Latent Dim")

    stds = stds.astype('str')
    means = means.astype('str')

    for i in range(stds.shape[0]):
        for j in range(stds.shape[1]):
            means[i][j] = "%.2f" % float(means[i][j])
            stds[i][j] = " ± %.2f" % float(stds[i][j])
    accuracies = np.char.add(means, stds)

    latent_col = np.array([l for l in latent_dims]).reshape(len(latent_dims), 1)
    formattedComps = np.hstack((latent_col, accuracies.T))
    df = pd.DataFrame(formattedComps, columns=(["Latent Dim"] + model_names))
    print(df.to_latex(index=False))

plt.xlabel("Dimensionality")
plt.ylabel("Testing Accuracy")
plt.xticks(range(latent_dims[0], latent_dims[-1]+1))
# plt.yticks(list(range(0, 109, 10)))
plt.savefig("%s_%s_accuracies.png" % (comparison_type, experiment_name))
