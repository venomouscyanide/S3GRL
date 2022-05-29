import argparse


def parse_hyper_tuner_results(file_name):
    with open(file_name, 'r') as hyper_file:
        results = hyper_file.readlines()

    max_lines = len(results)
    index = 0

    parsed_results = {
        "USAir": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "NS": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "Power": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "Celegans": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "Router": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "PB": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "Ecoli": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        },
        "Yeast": {
            "AUC": [],
            "AP": [],
            "Average Time": []
        }
    }
    while index < max_lines:
        line = results[index]
        if line.startswith("Command line input:"):
            dataset = line.split("--dataset")[-1].split("--hidden_channels")[0].strip()
            while 1:
                index += 1
                line = results[index]
                if line.startswith("All runs:"):
                    index += 2
                    line = results[index]
                    auc_score = float(line.split("Highest Test: ")[-1].split("± nan")[0].strip())

                    if dataset == 'Ecoli':
                        # TODO: ideally only this block is needed. Two verisons of the script was run
                        #  causing the need for this distinction
                        index += 8
                        line = results[index]
                        ap_score = float(line.split("Highest Test: ")[-1].split("± nan")[0].strip())

                        index += 8
                        line = results[index]
                        time_taken = float(
                            line.split("Time taken for run with ")[-1].split(" seconds")[0].split(": ")[-1].strip())
                    else:
                        index += 5
                        line = results[index]
                        ap_score = float(line.split("Highest Test: ")[-1].split("± nan")[0].strip())

                        index += 5
                        line = results[index]
                        time_taken = float(
                            line.split("Time taken for run with ")[-1].split(" seconds")[0].split(": ")[-1].strip())

                    parsed_results[dataset]['AUC'].append(auc_score)
                    parsed_results[dataset]['AP'].append(ap_score)
                    parsed_results[dataset]['Average Time'].append(time_taken)

                    index += 1
                    break
        index += 1
    print("Done reading file")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read the results from hyperparameter logs to get data to run plot.py"
    )
    parser.add_argument('--file_name', type=str)
    args = parser.parse_args()

    parse_hyper_tuner_results(args.file_name)
