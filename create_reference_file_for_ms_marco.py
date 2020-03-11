
import argparse
import json

def create_reference_file(input_file, ref_file):
    data = json.load(open(input_file, 'r'))
    num_questions = len(data['query'])

    with open(ref_file, 'w') as fout:
        for i in range(num_questions):
            qid = str(i)
            query = data['query'][qid]
            passages = []
            answers = data['answers'][qid]
            ret = {"query_id": qid, "answers": answers}
            fout.write(json.dumps(ret) + "\n")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--reference_file", default=None, type=str, required=True)
    args = parser.parse_args()

    create_reference_file(args.input_file, args.reference_file)

