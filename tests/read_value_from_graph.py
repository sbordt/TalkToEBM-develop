""" Test whether the LLM is able to read values from a graph.
"""
import guidance
import numpy as np

import t2ebm
from t2ebm import graphs
from t2ebm import prompts
import test_util


def graph_mean(graph, x_val):
    """Returns the mean of the graph at x_val."""
    # find the bin that x_val is in
    for idx, x_bin in enumerate(graph.x_vals):
        if x_val >= x_bin[0] and x_val <= x_bin[1]:
            bin_index = idx
            break
    # return the mean of that bin
    return graph.scores[bin_index]


def read_value_from_graph(llm, ebm, feature_name, x_val=None):
    feature_index = ebm.feature_names.index(feature_name)
    graph = graphs.extract_graph(ebm, feature_index)
    prompt = prompts.describe_graph(
        graphs.graph_to_text(graph, max_tokens=3000), include_assistant_response=True
    )
    # if x_val is not provided, choose a random bin and then sample a random value from within that bin
    if x_val is None:
        num_bins = len(graph.x_vals)
        bin_index = np.random.randint(num_bins)
        bin = graph.x_vals[bin_index]
        x_val = np.random.uniform(bin[0], bin[1])
    # the actual value of the graph at x_value
    y_val = graph_mean(graph, x_val)
    # the prompt
    prompt += (
        "\n\n{{#user~}}\nThanks. What is the mean value of the graph at "
        + f"{x_val:.4f}"
        + "?\n{{~/user}}\n\n"
    )
    prompt += """{{#assistant~}}{{gen 'value' temperature=0.7 max_tokens=100}}{{~/assistant}}\n\n"""
    print(guidance(prompt, llm=llm))
    # our prompts use guidance, and this is a nice way to print them
    response = guidance(prompt, llm=llm)()
    return x_val, y_val, response["value"]


if __name__ == "__main__":
    llm = test_util.openai_setup_gpt4()
    ebm = test_util.get_ebm(test_util.SPACESHIP_TITANIC)
    print(read_value_from_graph(llm, ebm, "Spa"))
