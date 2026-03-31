from evaluate import load
import json



def evaluator(predictions: list, references: list):
    """
    This function can be used to assess a whole file or individual sentences for their rouge, meteor and bertscore values.  
    The results will be written into a text file afterwards.  

    :param predictions: list
    :param references: list

    :Returns: none
    """

    rouge = load("rouge")
    meteor = load("meteor")
    bertscore = load("bertscore")

    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )

    meteor_results = meteor.compute(
        predictions=predictions,
        references=references
    )

    bertscore_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )

    with open("evaluation_results.txt", "a") as pap:
        pap.write("Rouge Results: ")
        json.dump(rouge_results, pap)
        pap.write("\n")
        pap.write("Meteor Results: ")
        json.dump(meteor_results, pap)
        pap.write("\n")
        pap.write("BertScore Results: ")
        json.dump(bertscore_results, pap)



if __name__ == "__main__":

    pred = ["In what form does Non-Hodgkin's lymphoma most frequently manifest in the orbit?"]
    ref = ["What is the most prevalent type of orbital cancer classified as a non-Hodgkin's lymphoma?"]

    evaluator(pred, ref)



# check all of the paraphrases if they are malformed 
# find some way to automate this 
# get some kind of mapping going so that one can filter sentences below a certain threshold out?

