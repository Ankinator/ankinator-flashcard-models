
def build_synthetic_model_outputs():
    model_outs = [
        (1, ["What is the capital of France?"]),
        (2, ["How do plants obtain energy?"]),
        (3, ["Are dogs considered mammals?"]),
        (4, ["What are the symptoms of COVID-19?"]),
        (5, ["Why is exercise important for a healthy lifestyle?"])
    ]
    refences = [
        (1, ["Which city is the capital of France?"]),
        (2, ["What is the source of energy for plants?"]),
        (3, ["Do cats fall under the category of mammals?"]),
        (4, ["Can you list the signs of COVID-19?"]),
        (5, ["What are the benefits of incorporating exercise into a daily routine?"])
    ]
    return model_outs, refences