import sys
import csv
from predict import predict

def avg(pli, store_path='submission_avg.csv'):
    results = list()
    for p in pli:
       res = predict(p, store_path='/dev/null')
       results.append(res)
    result = list()
    id_li = [e[0] for e in results[0]]
    for i, out_id in enumerate(id_li):
        single_avg = sum([res[i][1] for res in results]) / len(results)
        result.append([out_id, single_avg])
    with open(store_path, 'w', newline='') as fout:
        csv_writer = csv.writer(fout)
        csv_writer.writerow(['id', 'failure'])
        csv_writer.writerows(result)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f"Usage: {sys.argv[0]} <path to model1> [<path to model2> ...]")
    else:
        avg(sys.argv[1:])
