from math_verify import verify, parse

def math_eval(prediction, ground_truth):
    try:
        prediction = parse(prediction)
        ground_truth = parse(ground_truth)
        return int(verify(ground_truth, prediction))
    except Exception as e:
        print(f"Error: {e}")
        pass

    return 0
